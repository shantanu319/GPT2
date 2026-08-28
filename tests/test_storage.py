import os
from types import SimpleNamespace

import pytest

from vultr import storage


class FakeS3:
    """Enough of the boto3 client for the sync logic: sizes in, sizes out."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.uploaded, self.downloaded, self.buckets = [], [], []

    def create_bucket(self, Bucket):
        self.buckets.append(Bucket)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        # Serve one key per page so pagination is actually exercised.
        start = keys.index(ContinuationToken) if ContinuationToken else 0
        page = keys[start:start + 1]
        more = start + 1 < len(keys)
        return {"Contents": [{"Key": k, "Size": self.objects[k]} for k in page],
                "IsTruncated": more,
                **({"NextContinuationToken": keys[start + 1]} if more else {})}

    def upload_file(self, path, bucket, key):
        self.uploaded.append(key)
        self.objects[key] = os.path.getsize(path)

    def download_file(self, bucket, key, path):
        self.downloaded.append(key)
        with open(path, 'wb') as handle:
            handle.write(b'\0' * self.objects[key])


def _corpus(tmp_path, sizes):
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(b'\0' * size)
    return str(tmp_path)


def test_upload_skips_objects_already_the_same_size(tmp_path):
    local = _corpus(tmp_path, {'train_00000.bin': 10, 'train_00001.bin': 20})
    client = FakeS3({'run/train_00000.bin': 10})
    assert storage.upload_dir(client, local, 'run') == 1
    assert client.uploaded == ['run/train_00001.bin']


def test_upload_resends_a_truncated_object(tmp_path):
    local = _corpus(tmp_path, {'train_00000.bin': 10})
    client = FakeS3({'run/train_00000.bin': 4})   # interrupted mid-transfer
    assert storage.upload_dir(client, local, 'run') == 1


def test_upload_ignores_subdirectories(tmp_path):
    (tmp_path / 'nested').mkdir()
    local = _corpus(tmp_path, {'train_00000.bin': 10})
    client = FakeS3()
    assert storage.upload_dir(client, local, 'run') == 1


def test_download_skips_local_files_already_the_same_size(tmp_path):
    dest = tmp_path / 'down'
    dest.mkdir()
    (dest / 'train_00000.bin').write_bytes(b'\0' * 10)
    client = FakeS3({'run/train_00000.bin': 10, 'run/train_00001.bin': 20})
    assert storage.download_dir(client, 'run', str(dest)) == 1
    assert client.downloaded == ['run/train_00001.bin']
    assert (dest / 'train_00001.bin').stat().st_size == 20


def test_download_replaces_a_partial_local_file(tmp_path):
    dest = tmp_path / 'down'
    dest.mkdir()
    (dest / 'train_00000.bin').write_bytes(b'\0' * 3)
    client = FakeS3({'run/train_00000.bin': 10})
    assert storage.download_dir(client, 'run', str(dest)) == 1
    assert (dest / 'train_00000.bin').stat().st_size == 10


def test_remote_sizes_walks_every_page():
    client = FakeS3({f'run/shard_{i}.bin': i for i in range(5)})
    assert len(storage.remote_sizes(client, 'run')) == 5


def test_cluster_choice_prefers_the_training_region_then_amsterdam():
    clusters = [
        {'id': 2, 'region': 'ewr', 'deploy': 'yes'},
        {'id': 12, 'region': 'ams', 'deploy': 'yes'},
        {'id': 14, 'region': 'lhr', 'deploy': 'yes'},
    ]
    assert storage.select_cluster(clusters, 'lhr')['id'] == 14
    # object storage has no fra cluster, and the A100 plan deploys in fra
    assert storage.select_cluster(clusters, 'fra')['id'] == 12
    assert storage.select_cluster(clusters)['id'] == 12


def test_cluster_choice_ignores_undeployable_clusters():
    clusters = [{'id': 1, 'region': 'ams', 'deploy': 'no'},
                {'id': 2, 'region': 'ewr', 'deploy': 'yes'}]
    assert storage.select_cluster(clusters, 'ams')['id'] == 2
    with pytest.raises(RuntimeError, match="no deployable"):
        storage.select_cluster([{'id': 1, 'region': 'ams', 'deploy': 'no'}])


def test_subscription_waits_for_its_keys(monkeypatch):
    """A fresh subscription is created without credentials; polling until the
    keys appear is what makes the upload that follows work."""
    states = [
        {'object_storages': []},
        {'clusters': [{'id': 12, 'region': 'ams', 'deploy': 'yes',
                       'hostname': 'ams1.vultrobjects.com'}]},
        {'object_storage': {'id': 'sub-1', 'label': 'mot'}},
        {'object_storage': {'id': 'sub-1', 'label': 'mot'}},
        {'object_storage': {'id': 'sub-1', 'label': 'mot',
                            's3_access_key': 'AK', 's3_secret_key': 'SK',
                            's3_hostname': 'ams1.vultrobjects.com'}},
    ]
    api = SimpleNamespace(request=lambda *a, **k: states.pop(0))
    monkeypatch.setattr(storage.time, 'sleep', lambda _: None)
    sub = storage.ensure_subscription(api, label='mot')
    assert sub['s3_access_key'] == 'AK'


def test_subscription_reuses_an_existing_label(monkeypatch):
    api = SimpleNamespace(request=lambda *a, **k: {'object_storages': [
        {'id': 'sub-1', 'label': 'mot', 's3_access_key': 'AK'}]})
    assert storage.ensure_subscription(api, label='mot')['id'] == 'sub-1'


def test_env_credentials_name_what_is_missing(monkeypatch):
    for name in ("VULTR_S3_HOSTNAME", "VULTR_S3_ACCESS_KEY", "VULTR_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="VULTR_S3_HOSTNAME"):
        storage.subscription_from_env()
    monkeypatch.setenv("VULTR_S3_HOSTNAME", "ams1.vultrobjects.com")
    monkeypatch.setenv("VULTR_S3_ACCESS_KEY", "AK")
    with pytest.raises(RuntimeError, match="VULTR_S3_SECRET_KEY"):
        storage.subscription_from_env()
    monkeypatch.setenv("VULTR_S3_SECRET_KEY", "SK")
    assert storage.subscription_from_env()["s3_access_key"] == "AK"
