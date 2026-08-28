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


def _env_args(**kw):
    values = {"from_env": True, "label": "mot", "region": None, "workers": 2,
              "data_dir": ".", "prefix": "corpus", "max_shards": 0}
    values.update(kw)
    return SimpleNamespace(**values)


def test_upload_from_env_never_needs_the_account_api_key(monkeypatch, tmp_path):
    """The prep box holds only the S3 keys — asking for VULTR_API_KEY there is
    both a failure and a key we deliberately do not ship to a throwaway box."""
    monkeypatch.setattr(storage, "client_from_env",
                        lambda: pytest.fail("up must not reach for the account API"))
    monkeypatch.setenv("VULTR_S3_HOSTNAME", "ams1.vultrobjects.com")
    monkeypatch.setenv("VULTR_S3_ACCESS_KEY", "AK")
    monkeypatch.setenv("VULTR_S3_SECRET_KEY", "SK")
    client = FakeS3()
    monkeypatch.setattr(storage, "client_for", lambda sub: client)
    _corpus(tmp_path, {"train_00000.bin": 12})
    storage.up(_env_args(data_dir=str(tmp_path)))
    assert client.uploaded == ["corpus/train_00000.bin"]


def test_download_from_env_never_needs_the_account_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "client_from_env",
                        lambda: pytest.fail("down must not reach for the account API"))
    for name, value in (("VULTR_S3_HOSTNAME", "h"), ("VULTR_S3_ACCESS_KEY", "AK"),
                        ("VULTR_S3_SECRET_KEY", "SK")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(storage, "client_for",
                        lambda sub: FakeS3({"corpus/train_00000.bin": 5}))
    storage.down(_env_args(data_dir=str(tmp_path / "out")))
    assert (tmp_path / "out" / "train_00000.bin").stat().st_size == 5


def test_keep_shards_caps_train_shards_but_keeps_everything_else():
    """A 40B corpus is 80 shards; a run sized for 25B wants the first 50 and
    still needs the tokenizer, the holdouts, and the manifest."""
    keys = ["c/test.bin", "c/tokenizer.json", "c/train_00000.bin",
            "c/train_00001.bin", "c/train_00002.bin", "c/train_manifest.json",
            "c/val.bin"]
    assert storage.keep_shards(keys, 0) == keys
    kept = storage.keep_shards(keys, 2)
    assert "c/train_00002.bin" not in kept
    assert kept == ["c/test.bin", "c/tokenizer.json", "c/train_00000.bin",
                    "c/train_00001.bin", "c/train_manifest.json", "c/val.bin"]


def test_download_honours_the_shard_cap(monkeypatch, tmp_path):
    for name, value in (("VULTR_S3_HOSTNAME", "h"), ("VULTR_S3_ACCESS_KEY", "AK"),
                        ("VULTR_S3_SECRET_KEY", "SK")):
        monkeypatch.setenv(name, value)
    client = FakeS3({"corpus/train_00000.bin": 5, "corpus/train_00001.bin": 5,
                     "corpus/train_00002.bin": 5, "corpus/tokenizer.json": 5})
    monkeypatch.setattr(storage, "client_for", lambda sub: client)
    storage.down(_env_args(data_dir=str(tmp_path / "out"), max_shards=2))
    got = sorted(os.path.basename(p) for p in os.listdir(tmp_path / "out"))
    assert got == ["tokenizer.json", "train_00000.bin", "train_00001.bin"]


def test_prep_sizes_the_box_by_how_long_the_job_is():
    """Cheapest-$/hr alone picked 6 cores and 14.5h for a 40B prep, when the
    32-core plan costs the same total and finishes in 2.7h."""
    from vultr.prep import required_vcpu
    assert required_vcpu(20_000) == 1            # the smoke box is unchanged
    assert required_vcpu(31_000_000) == 18       # -> cheapest >=18-core plan
    assert required_vcpu(31_000_000, target_hours=2) == 9


def test_prep_passes_a_core_floor_to_provisioning(monkeypatch):
    """The floor is useless if prep never sets it."""
    from vultr import prep as prep_mod
    seen = {}

    def fake_provision(args, bootstrap_instance=False):
        seen.update(min_vcpu=args.min_vcpu, min_disk=args.min_disk)
        raise RuntimeError("stop here")

    monkeypatch.setattr(prep_mod, "client_from_env", lambda: None)
    monkeypatch.setattr(prep_mod, "ensure_subscription",
                        lambda *a, **k: {"region": "ams", "s3_hostname": "h",
                                         "s3_access_key": "AK", "s3_secret_key": "SK"})
    monkeypatch.setattr(prep_mod, "provision", fake_provision)
    args = SimpleNamespace(max_train_docs=31_000_000, disk=0, vcpu=0, keep=False,
                           region=None, label_storage="mot", prefix="c")
    with pytest.raises(RuntimeError, match="stop here"):
        prep_mod.prep(args)
    assert seen == {"min_vcpu": 18, "min_disk": 347}


def test_follow_shards_ships_sealed_shards_before_prepare_finishes(monkeypatch, tmp_path):
    """Serialising tokenize then upload loses everything if the box dies at
    90%. Shards are os.replace'd into place, so they can go early; val and the
    manifest are still being written and must wait."""
    import json as _json
    (tmp_path / "train_00000.bin").write_bytes(b"x" * 10)
    (tmp_path / "val.bin").write_bytes(b"y" * 4)
    (tmp_path / "train_manifest.json").write_text(_json.dumps({"complete": False}))
    client = FakeS3({})

    def finish(_seconds):
        (tmp_path / "train_00001.bin").write_bytes(b"z" * 10)
        (tmp_path / "train_manifest.json").write_text(_json.dumps({"complete": True}))

    monkeypatch.setattr(storage.time, "sleep", finish)
    storage.follow_shards(client, str(tmp_path), "c", poll=0)
    assert client.uploaded[0] == "c/train_00000.bin", "the sealed shard goes first"
    assert "c/val.bin" not in client.uploaded[:1], "val is still being appended"
    assert set(client.uploaded) == {"c/train_00000.bin", "c/train_00001.bin",
                                    "c/val.bin", "c/train_manifest.json"}


def test_prep_overlaps_the_upload_with_tokenizing():
    from types import SimpleNamespace as NS
    from vultr.prep import build_script
    script = build_script(
        NS(prefix="corpus-40b", max_train_docs=31_000_000, shard_tokens=500_000_000,
           workers=8),
        {"s3_hostname": "h", "s3_access_key": "AK", "s3_secret_key": "SK"})
    up = script.index("storage up --from-env --follow")
    tok = script.index("pretrain.prepare")
    assert up < tok, "the uploader must start before tokenizing, not after"
    assert "wait $UPLOADER" in script
    assert script.index("wait $UPLOADER") > tok


def test_post_stage_builds_sft_and_dpo_instead_of_the_corpus():
    """Without this the A100 box at $11.92/hr streams smol-smoltalk and
    ultrafeedback from HuggingFace itself."""
    from types import SimpleNamespace as NS
    from vultr.prep import build_script
    sub = {"s3_hostname": "h", "s3_access_key": "AK", "s3_secret_key": "SK"}
    args = NS(prefix="corpus-40b", max_train_docs=31_000_000,
              shard_tokens=500_000_000, workers=8, stage="post")
    script = build_script(args, sub)
    assert "sft.sft_prepare" in script and "dpo.dpo_prepare" in script
    assert "pretrain.prepare" not in script, "post must not re-tokenize the corpus"
    # they only share tokenizer.json, so neither blocks the other
    assert script.index("sft.sft_prepare") < script.index("wait $SFT")
    assert script.index("dpo.dpo_prepare") < script.index("wait $SFT")
    assert script.index("storage up") > script.index("wait $DPO")
    assert "--prefix corpus-40b" in script, "artifacts land beside the corpus"


def test_post_stage_asks_for_a_small_box():
    """Both prepares are single-threaded; sizing by max_train_docs would rent
    a 24-core box to run two single-threaded jobs."""
    from vultr import prep as prep_mod
    seen = {}

    def fake_provision(args, bootstrap_instance=False):
        seen.update(min_vcpu=args.min_vcpu, min_disk=args.min_disk,
                    min_ram=args.min_ram)
        raise RuntimeError("stop here")

    import pytest as _pytest
    from types import SimpleNamespace as NS
    monkey = _pytest.MonkeyPatch()
    monkey.setattr(prep_mod, "client_from_env", lambda: None)
    monkey.setattr(prep_mod, "ensure_subscription",
                   lambda *a, **k: {"region": "ams", "s3_hostname": "h",
                                    "s3_access_key": "AK", "s3_secret_key": "SK"})
    monkey.setattr(prep_mod, "provision", fake_provision)
    args = NS(max_train_docs=31_000_000, disk=0, vcpu=0, keep=False, region=None,
              label_storage="mot", prefix="corpus-40b", stage="post")
    with _pytest.raises(RuntimeError, match="stop here"):
        prep_mod.prep(args)
    monkey.undo()
    assert seen == {"min_vcpu": 2, "min_disk": 40, "min_ram": 4096}


def test_sft_and_dpo_artifacts_survive_the_shard_cap():
    """--corpus-shards 50 must not strip the post-training data."""
    keys = ["c/train_00000.bin", "c/train_00001.bin", "c/sft_train.bin",
            "c/sft_train_mask.bin", "c/dpo_train.bin", "c/dpo_train_pairs.bin",
            "c/tokenizer.json"]
    kept = storage.keep_shards(keys, 1)
    assert "c/train_00001.bin" not in kept
    for name in ("sft_train.bin", "sft_train_mask.bin", "dpo_train.bin",
                 "dpo_train_pairs.bin", "tokenizer.json"):
        assert f"c/{name}" in kept, f"{name} must survive the cap"
