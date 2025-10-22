import argparse
import os
from typing import Any

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from torch_dataset import create_dataloader
from models.GCN import GCNClassifier
from models.GAT import GATClassifier


def assemble_graph_inputs(x: torch.Tensor, mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
    B, T, F = x.shape
    assert F % 3 == 0
    infer_nodes = F // 3
    if num_nodes is not None:
        assert infer_nodes == num_nodes
    N = infer_nodes
    # Mask padded frames
    x = x * mask.unsqueeze(-1)
    x_reshaped = x.view(B, T, N, 3)
    x_perm = x_reshaped.permute(0, 2, 1, 3).contiguous()
    x_nodes = x_perm.view(B, N, T * 3)
    return x_nodes


def build_adjacency(num_nodes: int, device: torch.device) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def evaluate(model: torch.nn.Module, loader, device: torch.device, A: torch.Tensor, num_nodes: int):
    model.eval()
    class_names = loader.dataset.class_names
    num_classes = len(class_names)
    per_class_total = [0 for _ in range(num_classes)]
    per_class_correct = [0 for _ in range(num_classes)]
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            x_nodes = assemble_graph_inputs(x, mask, num_nodes)
            logits = model(x_nodes, A)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            for t, p in zip(y.view(-1), preds.view(-1)):
                ti = int(t.item())
                per_class_total[ti] += 1
                if int(p.item()) == ti:
                    per_class_correct[ti] += 1
    acc = correct / max(1, total)
    return acc, per_class_correct, per_class_total, class_names


def plot_per_class_hist(correct_counts, total_counts, class_names, out_path: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping per-class histogram plot.")
        return
    wrong_counts = [t - c for c, t in zip(correct_counts, total_counts)]
    x = list(range(len(class_names)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, correct_counts, color='tab:blue', label='correct')
    plt.bar(x, wrong_counts, bottom=correct_counts, color='red', alpha=0.7, label='wrong')
    for i, w in enumerate(wrong_counts):
        if w > 0:
            y_pos = correct_counts[i] + w / 2.0
            plt.text(i, y_pos, str(w), ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    plt.xticks(x, class_names, rotation=30, ha='right')
    plt.ylabel('Samples')
    plt.title('Per-class correct (blue) and wrong (red) counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved per-class histogram: {out_path}")


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics", "data", "cmu_mocap"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--source_fps", type=float, default=120.0)
    p.add_argument("--target_hz", type=float, default=30.0)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--overlap_sec", type=float, default=0.5)
    p.add_argument("--min_tail_sec", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--model", choices=["gcn", "gat"], default="gcn")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ckpt", default=None, help="Path to model checkpoint (.pt). If omitted, will try checkpoints/last_checkpoint.txt")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_dataloader(
        root=args.root,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        source_fps=args.source_fps,
        target_hz=args.target_hz,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        min_tail_sec=args.min_tail_sec,
    )
    print(f"Test DataLoader ready: batches={len(loader)} (batch_size={args.batch_size})")

    sample = next(iter(loader))
    F = sample["x"].shape[-1]
    assert F % 3 == 0
    num_nodes = F // 3

    num_classes = len(loader.dataset.class_names)
    in_node_dim = int(args.window_sec * args.target_hz) * 3

    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        default_marker = os.path.join(os.path.dirname(__file__), "checkpoints", "last_checkpoint.txt")
        if os.path.isfile(default_marker):
            with open(default_marker, "r", encoding="utf-8") as f:
                ckpt_path = f.read().strip()
        else:
            ckpt_path = None

    if ckpt_path and os.path.isfile(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device)
        meta = payload.get("meta", {})
        mtype = meta.get("model", args.model)
        hidden_dim = int(meta.get("hidden_dim", args.hidden_dim))
        layers = int(meta.get("layers", args.layers))
        heads = int(meta.get("heads", args.heads))
        if mtype == "gcn":
            model = GCNClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=layers).to(device)
        else:
            model = GATClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=layers, num_heads=heads).to(device)
        model.load_state_dict(payload["state_dict"], strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        if args.model == "gcn":
            model = GCNClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.layers).to(device)
        else:
            model = GATClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.layers, num_heads=args.heads).to(device)

    A = build_adjacency(num_nodes, device)
    acc, pc_correct, pc_total, class_names = evaluate(model, loader, device, A, num_nodes)
    print(f"test_acc={acc:.4f}")
    # Print per-class summary
    for name, c, t in zip(class_names, pc_correct, pc_total):
        w = t - c
        rate = (c / t) if t > 0 else 0.0
        print(f"- {name}: total={t} correct={c} wrong={w} acc={rate:.3f}")

    # Save single figure with blue(correct) + red overlay(wrong)
    out_path = os.path.join(os.path.dirname(__file__), 'test_per_class_hist.png')
    plot_per_class_hist(pc_correct, pc_total, class_names, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


