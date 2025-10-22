import argparse
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from torch_dataset import create_dataloader
from models.GCN import GCNClassifier
from models.GAT import GATClassifier


def assemble_graph_inputs(x: torch.Tensor, mask: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert windowed motion [B, T, F] into node features [B, N, 3*T].
    Assumes F = 3 * N (3D for each joint). Flattens per-node time series.
    """
    B, T, F = x.shape
    assert F % 3 == 0, "Feature dim must be 3 * num_nodes"
    infer_nodes = F // 3
    if num_nodes is not None:
        assert infer_nodes == num_nodes, f"num_nodes mismatch: inferred {infer_nodes} vs set {num_nodes}"
    N = infer_nodes
    # Mask padded frames (ensure masked frames contribute zeros)
    x = x * mask.unsqueeze(-1)
    x_reshaped = x.view(B, T, N, 3)  # [B,T,N,3]
    x_perm = x_reshaped.permute(0, 2, 1, 3).contiguous()  # [B,N,T,3]
    x_nodes = x_perm.view(B, N, T * 3)  # [B,N,3*T]
    return x_nodes


def build_adjacency(num_nodes: int, device: torch.device) -> torch.Tensor:
    """Simple skeleton adjacency: chain or star can be replaced with true skeleton if available."""
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def train_one_epoch(model: torch.nn.Module, loader, device: torch.device, criterion: nn.Module, optimizer: optim.Optimizer, A: torch.Tensor, num_nodes: int) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch in loader:
        x = batch["x"].to(device)  # [B,T,F]
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        B = x.size(0)
        x_nodes = assemble_graph_inputs(x, mask, num_nodes)  # [B,N,3*T]
        logits = model(x_nodes, A)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * B
        preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
        seen += int(B)
    avg_loss = total_loss / max(1, seen)
    acc = correct / max(1, seen)
    return avg_loss, acc


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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    p.add_argument("--exp_name", default=None)
    p.add_argument("--interleave_test", action="store_true", help="Run test during training every --test_every epochs")
    p.add_argument("--test_every", type=int, default=10)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_dataloader(
        root=args.root,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        source_fps=args.source_fps,
        target_hz=args.target_hz,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        min_tail_sec=args.min_tail_sec,
    )
    print(f"Train DataLoader ready: batches={len(loader)} (batch_size={args.batch_size})")

    # Infer num_nodes from one batch
    sample = next(iter(loader))
    F = sample["x"].shape[-1]
    assert F % 3 == 0
    num_nodes = F // 3

    num_classes = len(loader.dataset.class_names)
    in_node_dim = int(args.window_sec * args.target_hz) * 3

    if args.model == "gcn":
        model = GCNClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.layers).to(device)
    else:
        model = GATClassifier(num_nodes=num_nodes, in_node_dim=in_node_dim, hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.layers, num_heads=args.heads).to(device)

    A = build_adjacency(num_nodes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare checkpointing
    os.makedirs(args.save_dir, exist_ok=True)
    exp_name = (
        args.exp_name
        if args.exp_name
        else f"{args.model}_hz{int(args.target_hz)}_win{int(args.window_sec)}_ov{int(args.overlap_sec*10)}_hid{args.hidden_dim}_lay{args.layers}"
    )
    best_path = os.path.join(args.save_dir, exp_name + "_best.pt")
    latest_marker = os.path.join(args.save_dir, "last_checkpoint.txt")

    # Optional interleaved test every k epochs (k handled in main by calling test)
    # Optional test loader for interleaved evaluation
    test_loader = None
    if args.interleave_test:
        test_loader = create_dataloader(
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

    def evaluate_current() -> float:
        if test_loader is None:
            return -1.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)
                x_nodes = assemble_graph_inputs(x, mask, num_nodes)
                logits = model(x_nodes, A)
                preds = logits.argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += int(y.numel())
        return correct / max(1, total)

    best_score = -1.0
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, device, criterion, optimizer, A, num_nodes)
        print(f"epoch={epoch+1} loss={loss:.4f} acc={acc:.4f}")

        current_score = acc
        did_test = False
        if args.interleave_test and ((epoch + 1) % max(1, args.test_every) == 0):
            # Save a checkpoint BEFORE running the test, with epoch tag in filename
            epoch_tag_path = os.path.join(
                args.save_dir,
                f"{exp_name}_model{epoch+1}.pt",
            )
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {
                    "model": args.model,
                    "num_nodes": num_nodes,
                    "in_node_dim": in_node_dim,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                    "num_classes": num_classes,
                    "epoch": epoch + 1,
                }
            }, epoch_tag_path)
            print(f"Saved epoch-tag checkpoint before test: {epoch_tag_path}")
            test_acc = evaluate_current()
            if test_acc >= 0.0:
                print(f"epoch={epoch+1} test_acc={test_acc:.4f}")
                current_score = test_acc
                did_test = True

        if current_score > best_score:
            best_score = current_score
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {
                    "model": args.model,
                    "num_nodes": num_nodes,
                    "in_node_dim": in_node_dim,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                    "num_classes": num_classes,
                }
            }, best_path)
            with open(latest_marker, "w", encoding="utf-8") as f:
                f.write(best_path)
            tag = "test" if did_test else "train"
            print(f"Saved best checkpoint: {best_path} ({tag}_score={best_score:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


