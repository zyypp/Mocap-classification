import argparse
import os
from typing import Any

from train import main as train_main
from test import main as test_main


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser(description="Main entry for CMU MoCap classification experiments")
    p.add_argument("--mode", choices=["train", "test", "train_and_test"], default="train_and_test")
    p.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics", "data", "cmu_mocap"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--source_fps", type=float, default=120.0)
    p.add_argument("--target_hz", type=float, default=30.0)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--overlap_sec", type=float, default=1.5)
    p.add_argument("--min_tail_sec", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    args, extra = p.parse_known_args(argv)

    # Forward to the respective script with shared args
    common = [
        "--root", args.root,
        "--batch_size", str(args.batch_size),
        "--source_fps", str(args.source_fps),
        "--target_hz", str(args.target_hz),
        "--window_sec", str(args.window_sec),
        "--overlap_sec", str(args.overlap_sec),
        "--min_tail_sec", str(args.min_tail_sec),
        "--num_workers", str(args.num_workers),
        "--epochs", str(args.epochs),
    ] + extra

    if args.mode == "train":
        return train_main(common)
    elif args.mode == "test":
        # Remove training-only flags before calling test
        test_args = list(common)
        while "--epochs" in test_args:
            j = test_args.index("--epochs")
            del test_args[j:j+2]
        # strip interleave flags if present
        while "--interleave_test" in test_args:
            test_args.remove("--interleave_test")
        if "--test_every" in test_args:
            n = test_args.index("--test_every")
            del test_args[n:n+2]
        return test_main(test_args)
    else:
        # Single continuous training run with internal interleaved testing every 10 epochs
        train_args = list(common)
        if "--interleave_test" not in train_args:
            train_args.append("--interleave_test")
        if "--test_every" in train_args:
            m = train_args.index("--test_every")
            train_args[m + 1] = "10"
        else:
            train_args.extend(["--test_every", "10"])
        rc = train_main(train_args)
        if rc != 0:
            return rc
        # Run a final test using the latest best checkpoint
        test_args = list(train_args)
        while "--epochs" in test_args:
            j = test_args.index("--epochs")
            del test_args[j:j+2]
        while "--interleave_test" in test_args:
            test_args.remove("--interleave_test")
        if "--test_every" in test_args:
            n = test_args.index("--test_every")
            del test_args[n:n+2]
        return test_main(test_args)


if __name__ == "__main__":
    raise SystemExit(main())


