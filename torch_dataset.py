import os
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader

from cmu_dataloader_stats import CMUMocapDataset, _segment_sequence_counts


class CMUWindowedDataset(Dataset):
    """
    Windowed CMU MoCap dataset for classification at graph-ready cadence.

    - Downsamples to target_hz (via integer step rounding from source_fps)
    - Segments into windows of window_sec with overlap_sec; stride = window - overlap
    - Keeps tail >= min_tail_sec as a masked sample; drops tails < min_tail_sec
    - Returns tensors: inputs [window_frames, feat_dim], mask [window_frames], label: int
    """

    def __init__(
        self,
        root: str,
        split: str,
        source_fps: float = 120.0,
        target_hz: float = 30.0,
        window_sec: float = 2.0,
        overlap_sec: float = 1.5,
        min_tail_sec: float = 1.0,
        cache_downsampled: bool = True,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = os.path.abspath(root)
        self.split = split
        self.source_fps = float(source_fps)
        self.target_hz = float(target_hz)
        self.window_sec = float(window_sec)
        self.overlap_sec = float(overlap_sec)
        self.min_tail_sec = float(min_tail_sec)
        self.cache_downsampled = bool(cache_downsampled)

        # Build index and classes
        self.base = CMUMocapDataset(self.root)
        # Merge walking_extra into walking at dataset index level
        self.class_names = [c for c in self.base.classes() if c != "walking_extra"]
        self.class_to_label: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}

        # Determine effective rate and window frames
        _, _, eff_hz = _segment_sequence_counts(1, self.source_fps, self.target_hz, self.window_sec, self.overlap_sec, self.min_tail_sec)
        self.effective_hz = eff_hz
        self.window_frames = max(1, int(round(self.window_sec * self.effective_hz)))

        # Pre-index all windows across files
        self._entries: List[Tuple[str, int, int, int]] = []  # (path, label, start_idx_ds, length)
        for cls in self.class_names:
            files = []
            files.extend(self.base.index.get(self.split, {}).get(cls, []))
            if cls == "walking":
                files.extend(self.base.index.get(self.split, {}).get("walking_extra", []))
            label = self.class_to_label[cls]
            for path in files:
                length_frames = _count_lines_quick(path)
                num, valid_frames, _ = _segment_sequence_counts(length_frames, self.source_fps, self.target_hz, self.window_sec, self.overlap_sec, self.min_tail_sec)
                if num == 0:
                    continue
                # Derive stride and tail
                step = max(1, int(round(self.source_fps / float(self.target_hz))))
                m = (length_frames + step - 1) // step
                overlap_frames = int(round(max(0.0, self.overlap_sec) * self.effective_hz))
                stride = max(1, self.window_frames - overlap_frames)
                full = 0
                if m >= self.window_frames:
                    full = 1 + (m - self.window_frames) // stride
                # Full windows
                for i in range(full):
                    start = i * stride
                    self._entries.append((path, label, start, self.window_frames))
                # Tail (if any)
                consumed = 0 if full == 0 else (self.window_frames + (full - 1) * stride)
                tail = max(0, m - consumed)
                import math
                min_tail_frames = int(math.ceil(self.min_tail_sec * self.effective_hz))
                if tail >= min_tail_frames and tail < self.window_frames:
                    self._entries.append((path, label, consumed, tail))

        # Optional cache for downsampled tensors
        self._cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        path, label, start_idx_ds, length = self._entries[idx]
        x_ds = self._load_downsampled(path)

        # x_ds: [M, F]; extract window [start:start+length], pad to window_frames
        end_idx = min(x_ds.shape[0], start_idx_ds + length)
        window = x_ds[start_idx_ds:end_idx, :]

        feat_dim = window.shape[1]
        if length < self.window_frames:
            pad_frames = self.window_frames - length
            pad = torch.zeros((pad_frames, feat_dim), dtype=window.dtype)
            window = torch.cat([window, pad], dim=0)

        # mask: 1 for valid, 0 for padded
        mask = torch.zeros((self.window_frames,), dtype=torch.float32)
        mask[:length] = 1.0

        meta = {
            "path": path,
            "class_name": self.class_names[label],
            "start_idx_ds": start_idx_ds,
            "valid_length": int(length),
        }
        return window, mask, int(label), meta

    def _load_downsampled(self, path: str) -> torch.Tensor:
        if self.cache_downsampled and path in self._cache:
            return self._cache[path]
        # Read and downsample by integer step
        step = max(1, int(round(self.source_fps / float(self.target_hz))))
        rows: List[List[float]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if step > 1 and (i % step) != 0:
                    continue
                s = line.strip()
                if not s:
                    continue
                parts = s.split(',')
                try:
                    vals = [float(x) for x in parts]
                except Exception:
                    # tolerate parse errors by skipping line
                    continue
                rows.append(vals)
        if not rows:
            x = torch.zeros((0, 0), dtype=torch.float32)
        else:
            x = torch.tensor(rows, dtype=torch.float32)
        if self.cache_downsampled:
            self._cache[path] = x
        return x


def _count_lines_quick(path: str) -> int:
    cnt = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line and line.strip():
                cnt += 1
    return cnt


def cmu_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]]) -> Dict[str, Any]:
    # windows are already fixed-length; just stack
    xs = torch.stack([b[0] for b in batch], dim=0)  # [B, T, F]
    masks = torch.stack([b[1] for b in batch], dim=0)  # [B, T]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)  # [B]
    metas = [b[3] for b in batch]
    return {"x": xs, "mask": masks, "y": labels, "meta": metas}


def create_dataloader(
    root: str,
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader:
    ds = CMUWindowedDataset(root=root, split=split, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=cmu_collate, drop_last=False)


