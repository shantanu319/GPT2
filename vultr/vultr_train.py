"""Provision Vultr GPU instances and run the MyOwnTransformer training pipeline."""
import argparse
import os
import sys

if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

from vultr.jobs import PULL_DIR, pipeline, pull, push, ssh
from vultr.lifecycle import (
    DEFAULT_PRIVATE_KEY, DEFAULT_PUBLIC_KEY, GPU_OS_ID, destroy, print_plans, provision, status,
)
from vultr.smoke import smoke


def add_instance_args(parser, min_vram=20):
    parser.add_argument("--plan", help="exact Vultr GPU plan ID; default selects cheapest")
    parser.add_argument("--region", help="Vultr region; default uses the plan's first region")
    parser.add_argument("--min-vram", type=int, default=min_vram)
    parser.add_argument("--os-id", type=int, default=GPU_OS_ID)
    parser.add_argument("--label", default="myowntransformer")
    parser.add_argument("--ssh-public-key", default=DEFAULT_PUBLIC_KEY)
    parser.add_argument("--ssh-private-key", default=DEFAULT_PRIVATE_KEY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("plans", help="list catalog on-demand GPU plans by price")
    command.add_argument("--min-vram", type=int, default=0)
    command.set_defaults(func=print_plans)

    command = commands.add_parser("create", help="create and bootstrap a GPU instance")
    add_instance_args(command)
    command.set_defaults(func=provision)

    commands.add_parser("push", help="rsync the repository to the instance").set_defaults(func=push)

    command = commands.add_parser("ssh", help="print SSH command or run a remote command")
    command.add_argument("command", nargs=argparse.REMAINDER)
    command.set_defaults(func=ssh)

    command = commands.add_parser("pipeline", help="run prepare, pretrain, SFT, and DPO detached")
    command.add_argument("--dir-name", default="vultr_run")
    command.add_argument("--sft-dir-name", default="vultr_run_sft")
    command.add_argument("--dpo-dir-name", default="vultr_run_dpo")
    command.add_argument("--max-train-docs", type=int, default=1_000_000)
    command.add_argument("--d-model", type=int, default=512)
    command.add_argument("--n-layers", type=int, default=30)
    command.add_argument("--heads", type=int, default=8)
    command.add_argument("--kv-heads", type=int, default=2)
    command.add_argument("--batchsize", type=int, default=128)
    command.add_argument("--seqlen", type=int, default=1024)
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--warmup-steps", type=int, default=1000)
    command.add_argument("--save-every", type=int, default=4000)
    command.add_argument("--val-every", type=int, default=2000)
    command.add_argument("--sft-epochs", type=int, default=1)
    command.add_argument("--dpo-epochs", type=int, default=2)
    command.set_defaults(func=pipeline)

    command = commands.add_parser("status", help="show instance and pipeline status")
    command.add_argument("--id", help="instance ID for recovery without a state file")
    command.set_defaults(func=status)

    command = commands.add_parser("pull", help="download checkpoints and logs")
    command.add_argument("--out", default=PULL_DIR)
    command.set_defaults(func=pull)

    command = commands.add_parser("destroy", help="destroy the instance and stop billing")
    command.add_argument("--id", help="instance ID for recovery without a state file")
    command.set_defaults(func=destroy)

    command = commands.add_parser(
        "smoke", help="run tiny training on the cheapest GPU/compute, then destroy it"
    )
    add_instance_args(command, min_vram=2)
    command.add_argument("--compute", action="store_true", help="use the cheapest viable shared CPU plan")
    command.add_argument("--keep", action="store_true")
    command.add_argument("--out", default=PULL_DIR)
    command.set_defaults(func=smoke)

    load_dotenv(".env.local")
    args = parser.parse_args()
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
