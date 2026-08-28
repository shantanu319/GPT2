"""Vultr Object Storage: park the tokenized corpus between machines.

Prep is download-bound, not compute-bound, so it belongs on a cheap CPU box
rather than on a preemptible GPU box billing $11.92/hr to wait on
HuggingFace. Shards live here in between, which also means a reclaimed box
re-downloads a corpus in minutes instead of rebuilding it from scratch.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from vultr.api import client_from_env

BUCKET = "myowntransformer"
# Object storage has no fra cluster and the 8x A100 plan deploys there, so
# Amsterdam is the shortest hop for the corpus the GPU box has to pull.
DEFAULT_REGION = "ams"


def select_cluster(clusters, region=None):
    """Prefer the training region, else the documented nearest one."""
    live = [c for c in clusters if c.get("deploy") == "yes"]
    for wanted in (region, DEFAULT_REGION):
        match = next((c for c in live if c["region"] == wanted), None)
        if match:
            return match
    if not live:
        raise RuntimeError("no deployable object-storage cluster")
    return live[0]


def ensure_subscription(api, label="myowntransformer", region=None, timeout=300):
    """Find or create the subscription and return it once its keys exist."""
    match = find_subscription(api, label)
    if match is None:
        cluster = select_cluster(
            api.request("GET", "/object-storage/clusters?per_page=100")["clusters"], region)
        print(f"creating object storage in {cluster['region']} ({cluster['hostname']})...")
        match = api.request("POST", "/object-storage", {
            "cluster_id": cluster["id"], "label": label,
        })["object_storage"]
    deadline = time.time() + timeout
    while not match.get("s3_access_key"):
        if time.time() > deadline:
            raise RuntimeError(f"object storage {match['id']} has no keys after {timeout}s")
        time.sleep(5)
        match = api.request("GET", f"/object-storage/{match['id']}")["object_storage"]
    return match


def find_subscription(api, label="myowntransformer"):
    """Look up without creating — destroy must never provision to delete."""
    existing = api.request("GET", "/object-storage?per_page=500").get("object_storages", [])
    return next((s for s in existing if s.get("label") == label), None)


def destroy_subscription(api, subscription):
    api.request("DELETE", f"/object-storage/{subscription['id']}")
    print(f"destroyed object storage {subscription['id']}; billing stopped")


def client_for(subscription):
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=f"https://{subscription['s3_hostname']}",
        aws_access_key_id=subscription["s3_access_key"],
        aws_secret_access_key=subscription["s3_secret_key"],
        region_name="us-east-1",
    )


def remote_sizes(client, prefix, bucket=BUCKET):
    """{key: size} for everything already under prefix."""
    sizes = {}
    token = None
    while True:
        page = client.list_objects_v2(
            **{"Bucket": bucket, "Prefix": prefix,
               **({"ContinuationToken": token} if token else {})})
        for item in page.get("Contents", []):
            sizes[item["Key"]] = item["Size"]
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated"):
            return sizes


def _transfer(jobs, workers, action):
    if not jobs:
        return 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(action, jobs))
    return len(jobs)


def ensure_bucket(client, bucket=BUCKET):
    """create_bucket errors once the bucket exists, which every upload after
    the first would hit."""
    try:
        client.create_bucket(Bucket=bucket)
    except Exception as error:
        if not any(code in str(error) for code in
                   ("BucketAlreadyOwnedByYou", "BucketAlreadyExists")):
            raise


def upload_dir(client, local_dir, prefix, bucket=BUCKET, workers=8):
    """Upload local_dir under prefix, skipping objects already the same size.

    Size is enough: shards are sealed atomically and never rewritten, so a
    matching size means a matching object.
    """
    ensure_bucket(client, bucket)
    present = remote_sizes(client, prefix, bucket)
    jobs = []
    for name in sorted(os.listdir(local_dir)):
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            continue
        key = f"{prefix}/{name}"
        if present.get(key) == os.path.getsize(path):
            continue
        jobs.append((path, key))
    sent = _transfer(jobs, workers,
                     lambda job: client.upload_file(job[0], bucket, job[1]))
    print(f"uploaded {sent} file(s) to s3://{bucket}/{prefix} "
          f"({len(present)} already matched)")
    return sent


def download_dir(client, prefix, local_dir, bucket=BUCKET, workers=8):
    """Pull prefix into local_dir, skipping local files already the same size."""
    os.makedirs(local_dir, exist_ok=True)
    jobs = []
    for key, size in sorted(remote_sizes(client, prefix, bucket).items()):
        path = os.path.join(local_dir, os.path.basename(key))
        if os.path.exists(path) and os.path.getsize(path) == size:
            continue
        jobs.append((key, path))
    got = _transfer(jobs, workers,
                    lambda job: client.download_file(bucket, job[0], job[1]))
    print(f"downloaded {got} file(s) from s3://{bucket}/{prefix} into {local_dir}")
    return got


def subscription_from_env():
    """Creds the training box gets by env, the way HF_TOKEN already travels."""
    missing = [name for name in ("VULTR_S3_HOSTNAME", "VULTR_S3_ACCESS_KEY",
                                 "VULTR_S3_SECRET_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"object storage needs {', '.join(missing)} in the environment")
    return {"s3_hostname": os.environ["VULTR_S3_HOSTNAME"],
            "s3_access_key": os.environ["VULTR_S3_ACCESS_KEY"],
            "s3_secret_key": os.environ["VULTR_S3_SECRET_KEY"]}


def up(args):
    subscription = ensure_subscription(client_from_env(), args.label, args.region)
    upload_dir(client_for(subscription), args.data_dir, args.prefix,
               workers=args.workers)
    print(f"endpoint {subscription['s3_hostname']} (keys in the Vultr console)")


def down(args):
    """Pull on the training box from env creds, or locally from the account."""
    subscription = (subscription_from_env() if args.from_env
                    else ensure_subscription(client_from_env(), args.label, args.region))
    download_dir(client_for(subscription), args.prefix, args.data_dir,
                 workers=args.workers)


def destroy(args):
    api = client_from_env()
    subscription = find_subscription(api, args.label)
    if subscription is None:
        print(f"no object storage labelled {args.label}")
        return
    destroy_subscription(api, subscription)


def add_arguments(parser, default_prefix="corpus"):
    parser.add_argument("--data-dir", default="data_cache/cosmopedia")
    parser.add_argument("--prefix", default=default_prefix)
    parser.add_argument("--label", default="myowntransformer")
    parser.add_argument("--region", default=None,
                        help="object-storage region; defaults to the nearest to fra")
    parser.add_argument("--workers", type=int, default=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("up", up), ("down", down), ("destroy", destroy)):
        command = commands.add_parser(name)
        add_arguments(command)
        command.add_argument("--from-env", action="store_true",
                             help="use VULTR_S3_* instead of the account API")
        command.set_defaults(func=handler)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
