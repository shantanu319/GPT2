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


CLUSTERS = [
    {'id': 6, 'region': 'ams', 'deploy': 'yes', 'hostname': 'ams1'},
    {'id': 12, 'region': 'ams', 'deploy': 'yes', 'hostname': 'ams2'},
    {'id': 14, 'region': 'lhr', 'deploy': 'yes', 'hostname': 'lhr1'},
]
TIERS = {
    6: [{'id': 2, 'sales_name': 'Standard', 'disk_gb_price': 0.018},
        {'id': 3, 'sales_name': 'Premium', 'disk_gb_price': 0.036}],
    12: [{'id': 4, 'sales_name': 'Performance', 'disk_gb_price': 0.05}],
    14: [{'id': 4, 'sales_name': 'Performance', 'disk_gb_price': 0.05}],
}


def _placement_api(clusters=None):
    def request(method, path, payload=None, auth=True):
        if path.endswith('/tiers'):
            return {'tiers': TIERS[int(path.split('/')[-2])]}
        return {'clusters': clusters if clusters is not None else CLUSTERS}
    return SimpleNamespace(request=request)


def test_candidate_clusters_prefer_the_region_then_amsterdam():
    assert [c['id'] for c in storage.candidate_clusters(CLUSTERS, 'lhr')] == [14]
    # object storage has no fra cluster, and the A100 plan deploys in fra
    assert [c['id'] for c in storage.candidate_clusters(CLUSTERS, 'fra')] == [6, 12]
    assert [c['id'] for c in storage.candidate_clusters(CLUSTERS)] == [6, 12]


def test_candidate_clusters_ignore_undeployable_ones():
    clusters = [{'id': 1, 'region': 'ams', 'deploy': 'no'},
                {'id': 2, 'region': 'ewr', 'deploy': 'yes'}]
    assert [c['id'] for c in storage.candidate_clusters(clusters, 'ams')] == [2]
    with pytest.raises(RuntimeError, match="no deployable"):
        storage.candidate_clusters([{'id': 1, 'region': 'ams', 'deploy': 'no'}])


def test_placement_picks_the_cheapest_tier_not_just_the_first_cluster():
    """Both Amsterdam clusters serve the region, but ams2 is ~3x the price."""
    cluster, tier = storage.select_placement(_placement_api(), 'fra')
    assert cluster['id'] == 6 and tier['id'] == 2
    assert tier['disk_gb_price'] == 0.018


def test_placement_raises_when_a_region_offers_no_tier():
    api = SimpleNamespace(request=lambda method, path, payload=None, auth=True:
                          {'tiers': []} if path.endswith('/tiers')
                          else {'clusters': CLUSTERS})
    with pytest.raises(RuntimeError, match="no object-storage tier"):
        storage.select_placement(api, 'fra')


def test_subscription_waits_for_its_keys(monkeypatch):
    """A fresh subscription is created without credentials; polling until the
    keys appear is what makes the upload that follows work."""
    keyed = {'id': 'sub-1', 'label': 'mot', 's3_access_key': 'AK',
             's3_secret_key': 'SK', 's3_hostname': 'ams1.vultrobjects.com'}
    polls = [{'object_storage': {'id': 'sub-1', 'label': 'mot'}},
             {'object_storage': {'id': 'sub-1', 'label': 'mot'}},
             {'object_storage': keyed}]
    created = []

    def request(method, path, payload=None, auth=True):
        if path.startswith('/object-storage/clusters'):
            return ({'tiers': TIERS[int(path.split('/')[-2])]} if path.endswith('/tiers')
                    else {'clusters': CLUSTERS})
        if method == 'POST':
            created.append(payload)
            return {'object_storage': {'id': 'sub-1', 'label': 'mot'}}
        if path == '/object-storage?per_page=500':
            return {'object_storages': []}
        return polls.pop(0)

    monkeypatch.setattr(storage.time, 'sleep', lambda _: None)
    sub = storage.ensure_subscription(SimpleNamespace(request=request), label='mot')
    assert sub['s3_access_key'] == 'AK'
    # the tier is required by the API; creating without one is a 400
    assert created[0]['tier_id'] == 2 and created[0]['cluster_id'] == 6


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


class ExistingBucket(FakeS3):
    def create_bucket(self, Bucket):
        raise Exception("An error occurred (BucketAlreadyOwnedByYou) when calling "
                        "the CreateBucket operation")


class BrokenBucket(FakeS3):
    def create_bucket(self, Bucket):
        raise Exception("An error occurred (AccessDenied) when calling CreateBucket")


def test_upload_tolerates_a_bucket_that_already_exists(tmp_path):
    """Every upload after the first hits this; it must not be fatal."""
    local = _corpus(tmp_path, {'train_00000.bin': 10})
    client = ExistingBucket()
    assert storage.upload_dir(client, local, 'run') == 1


def test_upload_still_raises_on_a_real_bucket_failure(tmp_path):
    local = _corpus(tmp_path, {'train_00000.bin': 10})
    with pytest.raises(Exception, match="AccessDenied"):
        storage.upload_dir(BrokenBucket(), local, 'run')


def test_destroy_never_creates_a_subscription_in_order_to_delete_it(monkeypatch):
    """Going through ensure_subscription would provision one when absent."""
    calls = []

    def request(method, path, payload=None, auth=True):
        calls.append((method, path))
        return {"object_storages": []}

    monkeypatch.setattr(storage, "client_from_env",
                        lambda: SimpleNamespace(request=request))
    storage.destroy(SimpleNamespace(label="mot", region=None))
    assert not any(method == "POST" for method, _ in calls)


def test_find_subscription_returns_none_when_absent():
    api = SimpleNamespace(request=lambda *a, **k: {"object_storages": [
        {"id": "sub-1", "label": "other"}]})
    assert storage.find_subscription(api, "mot") is None
    assert storage.find_subscription(api, "other")["id"] == "sub-1"


def test_prep_bootstrap_installs_zstandard():
    import inspect
    from vultr import prep
    assert "zstandard" in inspect.getsource(prep._bootstrap)
