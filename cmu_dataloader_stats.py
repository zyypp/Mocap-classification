import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from statistics import mean, median

## Avoid numpy to remove dependency

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except Exception:
    plt = None
    mticker = None


class CMUMocapDataset:
    """
    Loader for the pre-split CMU MoCap dataset prepared under data/cmu_mocap.

    Expected structure:
      root/
        train/
          class_a/*.txt
          class_b/*.txt
          ...
        test/
          class_a/*.txt
          class_b/*.txt

    Each .txt file is a sequence with rows as frames (comma-separated floats).
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.train_dir = os.path.join(self.root, "train")
        self.test_dir = os.path.join(self.root, "test")
        if not os.path.isdir(self.train_dir) or not os.path.isdir(self.test_dir):
            raise FileNotFoundError(
                f"Expected 'train' and 'test' under {self.root}, found train={os.path.isdir(self.train_dir)} test={os.path.isdir(self.test_dir)}"
            )
        self.index = self._build_index()

    def _build_index(self) -> Dict[str, Dict[str, List[str]]]:
        # Merge walking_extra into walking to form 8-class setup
        def canonical_class(name: str) -> str:
            return "walking" if name == "walking_extra" else name

        index: Dict[str, Dict[str, List[str]]] = {"train": {}, "test": {}}
        for split, base in (("train", self.train_dir), ("test", self.test_dir)):
            accum: Dict[str, List[str]] = {}
            for raw_cls in sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))):
                cls = canonical_class(raw_cls)
                cls_dir = os.path.join(base, raw_cls)
                files = [os.path.join(cls_dir, f) for f in sorted(os.listdir(cls_dir)) if f.endswith(".txt")]
                if not files:
                    continue
                if cls not in accum:
                    accum[cls] = []
                accum[cls].extend(files)
            # sort files per class for determinism
            for k, v in accum.items():
                index[split][k] = sorted(v)
        return index

    def classes(self) -> List[str]:
        names = set(self.index.get("train", {}).keys()) | set(self.index.get("test", {}).keys())
        return sorted(names)

    def total_samples(self) -> int:
        return sum(len(v) for split in self.index.values() for v in split.values())

    def counts_per_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for split in ("train", "test"):
            for cls, files in self.index.get(split, {}).items():
                counts[cls] = counts.get(cls, 0) + len(files)
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))

    def lengths_per_class_frames(self) -> Dict[str, List[int]]:
        lengths: Dict[str, List[int]] = {cls: [] for cls in self.classes()}
        for split in ("train", "test"):
            for cls, files in self.index.get(split, {}).items():
                for path in files:
                    n = _count_lines(path)
                    lengths[cls].append(n)
        return lengths


def _count_lines(path: str) -> int:
    # Fast line counter; ignores trailing blank lines
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line and line.strip():
                count += 1
    return count


def _print_stats(dataset: CMUMocapDataset, fps: Optional[float]) -> None:
    print(f"Dataset root: {dataset.root}")
    print(f"Splits: train classes={len(dataset.index['train'])}, test classes={len(dataset.index['test'])}")
    print(f"Total samples (train+test): {dataset.total_samples()}")
    train_total = sum(len(v) for v in dataset.index.get('train', {}).values())
    test_total = sum(len(v) for v in dataset.index.get('test', {}).values())
    print(f"Train samples: {train_total}")
    print(f"Test samples: {test_total}")

    lengths = dataset.lengths_per_class_frames()
    print()
    print("Per-class sequence length stats:")
    for cls in dataset.classes():
        vals = list(lengths.get(cls, []))
        if len(vals) == 0:
            continue
        frames_stats = (int(min(vals)), float(median(vals)), float(mean(vals)), int(max(vals)))
        total_frames = int(sum(vals))
        if fps and fps > 0:
            secs_list = [v / float(fps) for v in vals]
            secs_stats = (min(secs_list), float(median(secs_list)), float(mean(secs_list)), max(secs_list))
            total_secs = float(total_frames) / float(fps)
            print(
                f"- {cls}: count={len(vals)} frames[min/median/mean/max]={frames_stats} total_frames={total_frames} seconds[min/median/mean/max]=({secs_stats[0]:.2f}, {secs_stats[1]:.2f}, {secs_stats[2]:.2f}, {secs_stats[3]:.2f}) total_seconds={total_secs:.2f}"
            )
        else:
            print(f"- {cls}: count={len(vals)} frames[min/median/mean/max]={frames_stats} total_frames={total_frames}")


def _plot_histogram(counts: Dict[str, int], out_path: str, title: Optional[str] = None) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot histograms. Please install matplotlib.")
    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    x = list(range(len(classes)))

    plt.figure(figsize=(10, 5))
    plt.bar(x, values, color="tab:blue")
    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylabel("Number of sequences")
    plt.title(title or "CMU MoCap per-class sample counts")
    plt.tight_layout()
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    plt.savefig(out_abs, dpi=150)
    plt.close()
    print(f"Saved histogram: {out_abs}")


def _lengths_by_split(dataset: CMUMocapDataset) -> Dict[str, Dict[str, List[int]]]:
    """
    Return per-split (train/test) per-class raw sequence lengths in frames.
    lengths[split][class] -> List[int]
    """
    lengths: Dict[str, Dict[str, List[int]]] = {"train": {}, "test": {}}
    for split in ("train", "test"):
        for cls, files in dataset.index.get(split, {}).items():
            arr: List[int] = []
            for path in files:
                arr.append(_count_lines(path))
            lengths[split][cls] = arr
    return lengths


def _plot_boxplots_per_class(
    lengths_split: Dict[str, Dict[str, List[int]]],
    out_dir: str,
    title_prefix: Optional[str] = None,
    fps_for_seconds: Optional[float] = None,
    y_scale: str = "linear",
    y2_scale: str = "linear",
    linthresh: float = 100.0,
    log_minor_subs: Optional[List[float]] = None,
    show_minor_labels: bool = False,
) -> None:
    """
    Plot and save per-class boxplots of raw sequence lengths for each split.
    Saves two images: boxplot_train.png and boxplot_test.png under out_dir.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required to plot boxplots. Please install matplotlib.")
    out_abs_dir = os.path.abspath(out_dir)
    os.makedirs(out_abs_dir, exist_ok=True)

    for split in ("train", "test"):
        classes = sorted(lengths_split.get(split, {}).keys())
        data = [lengths_split[split][c] for c in classes if len(lengths_split[split][c]) > 0]
        classes = [c for c in classes if len(lengths_split[split][c]) > 0]
        if not classes:
            continue
        # Increase vertical physical size for clearer separation of upper whisker and Q3
        fig, ax = plt.subplots(figsize=(12, 10))
        # Fill boxes with a single color for better readability
        bp = ax.boxplot(data, labels=classes, showfliers=True, patch_artist=True)
        for box in bp.get('boxes', []):
            box.set(facecolor="#8ecae6", edgecolor="black", alpha=0.7)
        for whisker in bp.get('whiskers', []):
            whisker.set(color="black", linewidth=1.2)
        for cap in bp.get('caps', []):
            cap.set(color="black", linewidth=1.2)
        for median_line in bp.get('medians', []):
            median_line.set(color="#d95f02", linewidth=2.0)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel("Sequence length (frames)")

        # Apply y-axis scale for frames
        if y_scale == "symlog":
            ax.set_yscale("symlog", linthresh=max(1e-6, float(linthresh)))
        else:
            ax.set_yscale(y_scale)
        # Denser tick marks according to scale
        if mticker is not None:
            if y_scale == "linear":
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
                try:
                    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                except Exception:
                    pass
            elif y_scale == "log":
                subs = tuple(log_minor_subs) if log_minor_subs else (1.0, 2.0, 5.0)
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
                ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=100))
                ax.yaxis.set_minor_formatter(mticker.LogFormatter(base=10) if show_minor_labels else mticker.NullFormatter())
            elif y_scale == "symlog":
                subs = tuple(log_minor_subs) if log_minor_subs else (1.0, 2.0, 5.0)
                if hasattr(mticker, "SymmetricalLogLocator"):
                    ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10.0, linthresh=max(1e-6, float(linthresh))))
                    ax.yaxis.set_minor_locator(mticker.SymmetricalLogLocator(base=10.0, linthresh=max(1e-6, float(linthresh)), subs=subs))
                else:
                    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
                    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=100))
                ax.yaxis.set_minor_formatter(mticker.LogFormatter(base=10) if show_minor_labels else mticker.NullFormatter())

        # Annotate median values above each box
        y_min, y_max = ax.get_ylim()
        y_span = max(1.0, (y_max - y_min))
        offset = 0.02 * y_span
        medians = bp.get('medians', [])
        for i, line in enumerate(medians):
            x0, x1 = line.get_xdata()[0], line.get_xdata()[1]
            x_center = (x0 + x1) / 2.0
            y_val = line.get_ydata()[0]
            if y_scale in ("log", "symlog"):
                # multiplicative offset is more appropriate for log-like scales
                ax.text(x_center, y_val * 1.05, f"{int(round(y_val))}", ha="center", va="bottom", fontsize=9, color="#d95f02")
            else:
                ax.text(x_center, y_val + offset, f"{int(round(y_val))}", ha="center", va="bottom", fontsize=9, color="#d95f02")

        # Right-axis line plot: per-class median duration (seconds)
        if fps_for_seconds and fps_for_seconds > 0:
            median_secs = [float(median(lengths_split[split][c])) / float(fps_for_seconds) for c in classes]
            ax2 = ax.twinx()
            xs = list(range(1, len(classes) + 1))  # boxplot uses 1-based positions
            ax2.plot(xs, median_secs, color="#2a9d8f", marker="o", linewidth=2.0)
            ax2.set_ylabel("Median duration (s)")
            ax2.grid(False)
            # Apply scaling for the right axis
            if y2_scale == "symlog":
                ax2.set_yscale("symlog", linthresh=max(1e-6, float(linthresh)) / float(fps_for_seconds))
            else:
                ax2.set_yscale(y2_scale)
            # Denser ticks for right axis
            if mticker is not None:
                if y2_scale == "linear":
                    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
                    try:
                        ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                    except Exception:
                        pass
                elif y2_scale in ("log", "symlog"):
                    subs = tuple(log_minor_subs) if log_minor_subs else (1.0, 2.0, 5.0)
                    if y2_scale == "log":
                        ax2.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
                        ax2.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=100))
                    else:
                        if hasattr(mticker, "SymmetricalLogLocator"):
                            ax2.yaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10.0, linthresh=max(1e-6, float(linthresh)) / float(fps_for_seconds)))
                            ax2.yaxis.set_minor_locator(mticker.SymmetricalLogLocator(base=10.0, linthresh=max(1e-6, float(linthresh)) / float(fps_for_seconds), subs=subs))
                        else:
                            ax2.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
                            ax2.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=100))
                    ax2.yaxis.set_minor_formatter(mticker.LogFormatter(base=10) if show_minor_labels else mticker.NullFormatter())

        title = (title_prefix + " ") if title_prefix else ""
        title += f"Per-class raw lengths ({split})"
        ax.set_title(title)
        fig.tight_layout()
        out_path = os.path.join(out_abs_dir, f"boxplot_{split}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved boxplot: {out_path}")


def _segment_sequence_counts(length_frames: int, source_fps: float, target_hz: float, window_sec: float, overlap_sec: float, min_tail_sec: float) -> Tuple[int, int, float]:
    """
    Return (num_samples, valid_frames_total, effective_hz) after downsampling and windowing.
    - Downsample by nearest integer factor to approach target_hz.
    - Window length = window_sec; overlap = overlap_sec; stride = window - overlap.
    - Tail rule: if remaining >= min_tail_sec then keep one masked sample with valid frames=remaining; else drop.
    """
    if source_fps <= 0 or target_hz <= 0:
        raise ValueError("source_fps and target_hz must be > 0")
    import math
    step = max(1, int(round(source_fps / float(target_hz))))
    effective_hz = float(source_fps) / float(step)
    # number of frames after downsampling by picking every 'step'
    m = (length_frames + step - 1) // step  # ceil-like count of sampled frames
    window_frames = max(1, int(round(window_sec * effective_hz)))
    overlap_frames = int(round(max(0.0, overlap_sec) * effective_hz))
    stride = max(1, window_frames - overlap_frames)
    min_tail_frames = int(math.ceil(min_tail_sec * effective_hz))

    if m <= 0:
        return 0, 0, effective_hz

    full = 0
    if m >= window_frames:
        full = 1 + (m - window_frames) // stride
    consumed = 0 if full == 0 else (window_frames + (full - 1) * stride)
    tail = max(0, m - consumed)
    keep_tail = 1 if (tail >= min_tail_frames and tail < window_frames) else 0
    num_samples = full + keep_tail
    valid_frames_total = full * window_frames + (tail if keep_tail else 0)
    return num_samples, valid_frames_total, effective_hz


def compute_windowed_stats(dataset: CMUMocapDataset, source_fps: float, target_hz: float, window_sec: float, overlap_sec: float, min_tail_sec: float) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], float, int]:
    """
    Compute per-split (train/test) per-class sample counts and totals after windowing.
    Returns:
      counts[split][class] -> sample_count
      frames[split][class] -> total_valid_frames
      seconds[split][class] -> total_seconds (valid_frames/effective_hz)
      effective_hz -> actual sampling rate after integer downsample
      window_frames -> window size in frames at effective_hz
    """
    counts: Dict[str, Dict[str, int]] = {"train": {}, "test": {}}
    frames: Dict[str, Dict[str, int]] = {"train": {}, "test": {}}
    seconds: Dict[str, Dict[str, float]] = {"train": {}, "test": {}}

    # Determine effective_hz and window_frames once using a representative rate
    # We still recompute inside in case of edge rounding, but we'll report a single effective_hz
    _, _, effective_hz = _segment_sequence_counts(1, source_fps, target_hz, window_sec, overlap_sec, min_tail_sec)
    window_frames = max(1, int(round(window_sec * effective_hz)))

    for split in ("train", "test"):
        for cls, files in dataset.index.get(split, {}).items():
            total_samples = 0
            total_valid_frames = 0
            for path in files:
                n = _count_lines(path)
                num, valid_frames, _ = _segment_sequence_counts(n, source_fps, target_hz, window_sec, overlap_sec, min_tail_sec)
                total_samples += num
                total_valid_frames += valid_frames
            counts[split][cls] = total_samples
            frames[split][cls] = total_valid_frames
            seconds[split][cls] = (float(total_valid_frames) / effective_hz) if effective_hz > 0 else 0.0

    return counts, frames, seconds, effective_hz, window_frames


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CMU MoCap pre-split dataloader and statistics")
    parser.add_argument(
        "--root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics",
            "data",
            "cmu_mocap",
        ),
        help="Root of the pre-split CMU MoCap dataset (contains train/ and test/)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Frame rate to convert frames to seconds (optional)")
    parser.add_argument("--source_fps", type=float, default=120.0, help="Original data frame rate (Hz)")
    parser.add_argument("--target_hz", type=float, default=30.0, help="Target sampling rate (Hz)")
    parser.add_argument("--window_sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--overlap_sec", type=float, default=1.5, help="Window overlap in seconds")
    parser.add_argument("--min_tail_sec", type=float, default=1.0, help=">= this tail seconds kept as masked sample; else dropped")
    parser.add_argument(
        "--hist_out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmu_class_counts.png"),
        help="Output path for per-class count histogram PNG",
    )
    parser.add_argument(
        "--boxplot_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "boxplot"),
        help="Output directory to save per-class raw-length boxplots",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting the histogram")
    parser.add_argument("--y_scale", choices=["linear", "log", "symlog"], default="linear", help="Y-axis scale for frames (left axis)")
    parser.add_argument("--y2_scale", choices=["linear", "log", "symlog"], default="linear", help="Y-axis scale for durations (right axis)")
    parser.add_argument("--linthresh", type=float, default=100.0, help="linthresh for symlog scale (in frames on left axis)")
    parser.add_argument("--log_minor_subs", default="1,2,5", help="Comma list of minor tick multiples per decade for log/symlog (e.g., 1,2,5 or 1,2,3,4,5,6,7,8,9)")
    parser.add_argument("--show_minor_labels", action="store_true", help="Show labels for minor log ticks (may be crowded)")

    args = parser.parse_args(argv)

    ds = CMUMocapDataset(args.root)
    _print_stats(ds, args.fps)

    # Windowing summary for downstream graph models
    print()
    print("Windowing parameters:")
    print(f"- source_fps={args.source_fps}")
    print(f"- target_hz={args.target_hz}")
    print(f"- window_sec={args.window_sec}")
    print(f"- overlap_sec={args.overlap_sec}")
    print(f"- min_tail_sec={args.min_tail_sec}")

    counts_split, frames_split, seconds_split, effective_hz, window_frames = compute_windowed_stats(
        ds,
        source_fps=args.source_fps,
        target_hz=args.target_hz,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        min_tail_sec=args.min_tail_sec,
    )

    print()
    print(f"Effective sampling rate after downsampling: {effective_hz:.6f} Hz (window_frames={window_frames})")

    for split in ("train", "test"):
        print()
        print(f"[{split}] per-class windowed stats:")
        for cls in ds.classes():
            # Some classes may not exist in one split; default to 0
            c = counts_split.get(split, {}).get(cls, 0)
            f = frames_split.get(split, {}).get(cls, 0)
            s = seconds_split.get(split, {}).get(cls, 0.0)
            print(f"- {cls}: samples={c}, total_frames={f}, total_seconds={s:.2f}")

    # Combined counts for histogram (train+test)
    combined_counts: Dict[str, int] = {}
    for cls in ds.classes():
        combined_counts[cls] = counts_split.get("train", {}).get(cls, 0) + counts_split.get("test", {}).get(cls, 0)

    if not args.no_plot:
        # Combined
        _plot_histogram(combined_counts, args.hist_out, title="CMU MoCap per-class sample counts (windowed)")
        # Train/Test split-specific plots
        base, ext = os.path.splitext(args.hist_out)
        _plot_histogram(counts_split.get("train", {}), base + "_train" + ext, title="Train per-class sample counts (windowed)")
        _plot_histogram(counts_split.get("test", {}), base + "_test" + ext, title="Test per-class sample counts (windowed)")
        # Per-class raw-length boxplots (no windowing, original sequence lengths)
        lengths_split = _lengths_by_split(ds)
        # Parse minor subs
        try:
            subs = tuple(float(x) for x in (args.log_minor_subs.split(",") if args.log_minor_subs else []))
        except Exception:
            subs = (1.0, 2.0, 5.0)

        _plot_boxplots_per_class(
            lengths_split,
            args.boxplot_dir,
            title_prefix="CMU MoCap",
            fps_for_seconds=args.source_fps,
            y_scale=args.y_scale,
            y2_scale=args.y2_scale,
            linthresh=args.linthresh,
            log_minor_subs=list(subs),
            show_minor_labels=args.show_minor_labels,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


