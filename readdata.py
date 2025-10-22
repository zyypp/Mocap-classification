import os
import re
import sys
import math
import json
import glob
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None
    animation = None


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    channels: List[str]
    parent: Optional[int]
    children: List[int]


@dataclass
class BVH:
    joints: List[Joint]
    name_to_index: Dict[str, int]
    frames: int
    frame_time: float
    motion: np.ndarray  # shape: (frames, num_channels)
    root_index: int

    def channel_indices(self) -> Dict[int, Tuple[int, int]]:
        indices: Dict[int, Tuple[int, int]] = {}
        cursor = 0
        for idx, j in enumerate(self.joints):
            count = len(j.channels)
            if count:
                indices[idx] = (cursor, cursor + count)
            cursor += count
        return indices

    def skeleton_edges(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for i, j in enumerate(self.joints):
            for c in j.children:
                edges.append((i, c))
        return edges


# -------------------------------
# BVH parsing
# -------------------------------

_channel_re = re.compile(r"CHANNELS\s+(\d+)\s+(.+)")
_offset_re = re.compile(r"OFFSET\s+([-0-9eE.+]+)\s+([-0-9eE.+]+)\s+([-0-9eE.+]+)")
_joint_re = re.compile(r"(JOINT|ROOT)\s+(.+)")
_end_re = re.compile(r"End Site")
_frames_re = re.compile(r"Frames\s*:\s*(\d+)", re.IGNORECASE)
_frame_time_re = re.compile(r"Frame\s*Time\s*:\s*([-0-9eE.+]+)", re.IGNORECASE)


def _parse_channels(tokens: List[str]) -> List[str]:
    allowed = {
        "Xposition", "Yposition", "Zposition",
        "Xrotation", "Yrotation", "Zrotation",
    }
    channels: List[str] = []
    for t in tokens:
        tt = t.strip()
        if tt:
            if tt not in allowed:
                raise ValueError(f"Unsupported BVH channel: {tt}")
            channels.append(tt)
    return channels


def load_bvh(path: str) -> BVH:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    assert lines[0].strip().upper() == "HIERARCHY", "BVH missing HIERARCHY header"

    joints: List[Joint] = []
    name_to_index: Dict[str, int] = {}

    stack: List[int] = []
    i = 1
    root_index: Optional[int] = None

    # Parse hierarchy
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.upper() == "MOTION":
            i += 1
            break

        m_j = _joint_re.match(line)
        if m_j:
            kind, name = m_j.group(1), m_j.group(2).strip()
            # Expect '{' on next non-empty
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            assert lines[i].strip() == "{", f"BVH parse error: expected '{{' after {kind} {name}"

            # Defaults until seen
            offset = np.zeros(3, dtype=np.float64)
            channels: List[str] = []

            # Parent is top of stack (if any)
            parent = stack[-1] if stack else None

            # Create joint; children unknown yet
            idx = len(joints)
            joints.append(Joint(name=name, offset=offset, channels=channels, parent=parent, children=[]))
            name_to_index[name] = idx

            if parent is not None:
                joints[parent].children.append(idx)
            else:
                root_index = idx

            # Push current joint to stack
            stack.append(idx)
            i += 1
            continue

        if _end_re.match(line):
            # End Site block: define a terminal joint without channels
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            assert i < len(lines) and lines[i].strip() == "{", "BVH parse error: expected '{' after End Site"
            # Scan inside braces for an OFFSET line; tolerate blanks/format variations
            i += 1
            end_offset: Optional[np.ndarray] = None
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s == "}":
                    i += 1
                    break
                m_off = _offset_re.match(s)
                if m_off and end_offset is None:
                    end_offset = np.array([
                        float(m_off.group(1)),
                        float(m_off.group(2)),
                        float(m_off.group(3)),
                    ], dtype=np.float64)
                i += 1
            if end_offset is None:
                end_offset = np.zeros(3, dtype=np.float64)
            # Create synthetic name for end site under current parent if available; otherwise skip gracefully
            if stack:
                parent = stack[-1]
                idx = len(joints)
                end_name = f"{joints[parent].name}_end"
                joints.append(Joint(name=end_name, offset=end_offset, channels=[], parent=parent, children=[]))
                joints[parent].children.append(idx)
                name_to_index[end_name] = idx
            # if no parent, we already consumed the block; just continue
            continue

        if line == "}":
            # pop one scope; if stray '}' appears with empty stack, skip gracefully
            if stack:
                stack.pop()
            i += 1
            continue

        m_off = _offset_re.match(line)
        if m_off:
            assert stack, "OFFSET outside of joint scope"
            cur = joints[stack[-1]]
            cur.offset = np.array([float(m_off.group(1)), float(m_off.group(2)), float(m_off.group(3))], dtype=np.float64)
            i += 1
            continue

        m_ch = _channel_re.match(line)
        if m_ch:
            assert stack, "CHANNELS outside of joint scope"
            num = int(m_ch.group(1))
            toks = m_ch.group(2).split()
            channels = _parse_channels(toks)
            if len(channels) != num:
                raise ValueError("CHANNELS count mismatch")
            joints[stack[-1]].channels = channels
            i += 1
            continue

        # Unknown line in hierarchy; skip
        i += 1

    # Parse motion header
    assert i < len(lines), "BVH missing MOTION section"
    # Find 'Frames:' line by scanning forward (tolerate comments/whitespace)
    m_frames = None
    while i < len(lines):
        s = lines[i].strip()
        if s:
            m_frames = _frames_re.match(s)
            if m_frames:
                break
        i += 1
    assert m_frames, "BVH missing Frames line"
    frames = int(m_frames.group(1))
    i += 1

    # Find 'Frame Time:' line by scanning forward
    m_ft = None
    while i < len(lines):
        s = lines[i].strip()
        if s:
            m_ft = _frame_time_re.match(s)
            if m_ft:
                break
        i += 1
    assert m_ft, "BVH missing Frame Time line"
    frame_time = float(m_ft.group(1))
    i += 1

    # Parse motion data
    num_channels = sum(len(j.channels) for j in joints)
    motion = np.zeros((frames, num_channels), dtype=np.float64)
    row = 0
    while i < len(lines) and row < frames:
        line = lines[i].strip()
        if line:
            vals = [float(x) for x in line.split()]
            if len(vals) != num_channels:
                # Lines can be split; keep reading until we have enough
                parts = vals
                jdx = i + 1
                while len(parts) < num_channels and jdx < len(lines):
                    extra = [float(x) for x in lines[jdx].strip().split()]
                    parts.extend(extra)
                    jdx += 1
                i = jdx - 1
                vals = parts
            if len(vals) != num_channels:
                raise ValueError(f"Frame {row} channel count mismatch: got {len(vals)}, expected {num_channels}")
            motion[row, :] = np.array(vals, dtype=np.float64)
            row += 1
        i += 1

    if row != frames:
        raise ValueError(f"Expected {frames} frames, parsed {row}")

    assert root_index is not None, "No ROOT joint parsed"
    name_to_index = dict(name_to_index)

    return BVH(
        joints=joints,
        name_to_index=name_to_index,
        frames=frames,
        frame_time=frame_time,
        motion=motion,
        root_index=root_index,
    )


# -------------------------------
# Forward kinematics
# -------------------------------

def _deg2rad(x: np.ndarray) -> np.ndarray:
    return x * (math.pi / 180.0)


def _rotation_matrix(rx: float, ry: float, rz: float, order: List[str]) -> np.ndarray:
    # BVH rotations specify an order like Zrotation Xrotation Yrotation
    # We'll compose accordingly (right-multiply per channel order)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float64)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float64)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float64)

    mapping = {
        "Xrotation": Rx,
        "Yrotation": Ry,
        "Zrotation": Rz,
    }
    R = np.eye(3)
    for ch in order:
        if ch.endswith("rotation"):
            R = R @ mapping[ch]
    return R


def compute_global_positions(bvh: BVH, frame_index: int) -> np.ndarray:
    if frame_index < 0 or frame_index >= bvh.frames:
        raise IndexError("frame_index out of range")

    channel_ranges = bvh.channel_indices()

    global_positions = np.zeros((len(bvh.joints), 3), dtype=np.float64)
    global_rotations = [np.eye(3, dtype=np.float64) for _ in bvh.joints]

    def recurse(joint_index: int, parent_pos: np.ndarray, parent_rot: np.ndarray):
        j = bvh.joints[joint_index]
        # Determine local transform from channels
        local_pos = j.offset.copy()
        local_rot = np.eye(3, dtype=np.float64)

        if joint_index in channel_ranges:
            start, end = channel_ranges[joint_index]
            values = bvh.motion[frame_index, start:end]
            # Position channels (if any) are always in X/Y/Z order when present
            px = py = pz = 0.0
            rx = ry = rz = 0.0
            rotation_order: List[str] = []
            for ch, val in zip(j.channels, values):
                if ch == "Xposition":
                    px = val
                elif ch == "Yposition":
                    py = val
                elif ch == "Zposition":
                    pz = val
                elif ch.endswith("rotation"):
                    rotation_order.append(ch)
                    if ch == "Xrotation":
                        rx = val
                    elif ch == "Yrotation":
                        ry = val
                    elif ch == "Zrotation":
                        rz = val
            local_pos = local_pos + np.array([px, py, pz], dtype=np.float64)
            local_rot = _rotation_matrix(_deg2rad(rx), _deg2rad(ry), _deg2rad(rz), rotation_order)

        # Compose
        world_pos = parent_pos + parent_rot @ local_pos
        world_rot = parent_rot @ local_rot

        global_positions[joint_index] = world_pos
        global_rotations[joint_index] = world_rot

        for c in j.children:
            recurse(c, world_pos, world_rot)

    # Root parent is origin
    recurse(bvh.root_index, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
    return global_positions


# -------------------------------
# Utilities
# -------------------------------

def list_bvh_files(root: str = "/data/yanyu/GF-theory/Mocap/cmu-mocap/data") -> List[str]:
    pattern = os.path.join(root, "**", "*.bvh")
    return sorted(glob.glob(pattern, recursive=True))


# -------------------------------
# Visualization
# -------------------------------

def visualize_bvh(path: str, downsample: int = 1, max_frames: Optional[int] = None, title: Optional[str] = None, save: Optional[str] = None, fps: Optional[int] = None, dpi: int = 120, interactive: bool = False, export_frames: Optional[str] = None):
    if plt is None or animation is None:
        raise RuntimeError("matplotlib is required for visualization. Please install it.")

    bvh = load_bvh(path)
    edges = bvh.skeleton_edges()
    print(f"Loaded BVH: {path}")
    print(f"- Frames: {bvh.frames}, FrameTime: {bvh.frame_time:.6f}s ({(1.0/bvh.frame_time) if bvh.frame_time>0 else float('inf'):.2f} fps)")
    print(f"- Joints: {len(bvh.joints)}, Edges: {len(edges)}")

    step = max(1, int(downsample))
    frame_indices = list(range(0, bvh.frames, step))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    # Precompute positions for speed
    positions = [compute_global_positions(bvh, f) for f in frame_indices]

    # Determine bounds
    all_pts = np.concatenate(positions, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    if extent <= 0:
        extent = 1.0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title or os.path.basename(path))

    # Initialize lines per edge
    lines = []
    for (i, j) in edges:
        (ln,) = ax.plot([0, 0], [0, 0], [0, 0], color='tab:blue', lw=2)
        lines.append((ln, i, j))

    def set_equal_3d(ax_):
        ax_.set_xlim(center[0] - extent * 0.6, center[0] + extent * 0.6)
        ax_.set_ylim(center[1] - extent * 0.6, center[1] + extent * 0.6)
        ax_.set_zlim(center[2] - extent * 0.6, center[2] + extent * 0.6)
        ax_.set_xlabel('X')
        ax_.set_ylabel('Y')
        ax_.set_zlabel('Z')

    set_equal_3d(ax)

    # Frame overlay text (top-left)
    text_overlay = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        for ln, i, j in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        text_overlay.set_text("")
        return [ln for ln, _, _ in lines] + [text_overlay]

    def animate_func(k: int):
        pts = positions[k]
        for ln, i, j in lines:
            xi, yi, zi = pts[i]
            xj, yj, zj = pts[j]
            ln.set_data([xi, xj], [yi, yj])
            ln.set_3d_properties([zi, zj])
        text_overlay.set_text(f"frame: {frame_indices[k]}")
        return [ln for ln, _, _ in lines] + [text_overlay]

    if interactive and save is None:
        # Build an interactive slider to scrub frames
        plt.subplots_adjust(bottom=0.18)
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
        slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=len(positions) - 1, valinit=0, valfmt='%0.0f')

        # Draw frame 0 initially
        def draw_frame(k: int):
            pts = positions[k]
            for ln, i, j in lines:
                xi, yi, zi = pts[i]
                xj, yj, zj = pts[j]
                ln.set_data([xi, xj], [yi, yj])
                ln.set_3d_properties([zi, zj])
            text_overlay.set_text(f"frame: {frame_indices[k]}")
            fig.canvas.draw_idle()

        draw_frame(0)

        def on_slider(val):
            draw_frame(int(val))
        slider.on_changed(on_slider)

        # Keyboard: left/right to step, space to play/pause
        playing = {"flag": False}

        def on_key(event):
            if event.key in ('left',):
                k = int(slider.val) - 1
                if k < 0:
                    k = 0
                slider.set_val(k)
            elif event.key in ('right',):
                k = int(slider.val) + 1
                if k > len(positions) - 1:
                    k = len(positions) - 1
                slider.set_val(k)
            elif event.key in (' ', 'space'):
                playing["flag"] = not playing["flag"]

        fig.canvas.mpl_connect('key_press_event', on_key)

        def timer_step(_):
            if not playing["flag"]:
                return
            k = int(slider.val) + 1
            if k >= len(positions):
                k = 0
            slider.set_val(k)

        timer = fig.canvas.new_timer(interval=int(bvh.frame_time * 1000.0 * step))
        timer.add_callback(timer_step, None)
        timer.start()

        plt.tight_layout()
        plt.show()
        return

    ani = animation.FuncAnimation(fig, animate_func, init_func=init, frames=len(positions), interval=bvh.frame_time * 1000.0 * step, blit=True)

    plt.tight_layout()

    # Export per-frame PNGs
    if export_frames:
        out_dir = os.path.abspath(export_frames)
        os.makedirs(out_dir, exist_ok=True)
        # Render once per frame using the animator function
        print(f"Exporting {len(positions)} frames to: {out_dir}")
        for k in range(len(positions)):
            animate_func(k)
            fig.savefig(os.path.join(out_dir, f"frame_{frame_indices[k]:06d}.png"), dpi=dpi)
        print("Export frames done.")

    if save:
        out_path = os.path.abspath(save)
        ext = os.path.splitext(out_path)[1].lower()
        try:
            if ext in (".mp4", ".m4v", ".mov"):
                # Ensure ffmpeg is available, try PATH then imageio-ffmpeg
                ffmpeg_exec = shutil.which("ffmpeg")
                if ffmpeg_exec is None:
                    try:
                        import imageio_ffmpeg  # type: ignore
                        ffmpeg_exec = imageio_ffmpeg.get_ffmpeg_exe()
                    except Exception:
                        ffmpeg_exec = None
                if ffmpeg_exec is None:
                    raise RuntimeError(
                        "ffmpeg not found. Install via 'sudo apt-get install ffmpeg' or 'conda install -c conda-forge ffmpeg', "
                        "or save as GIF with --save output.gif (requires pillow)."
                    )
                try:
                    from matplotlib import rcParams
                    rcParams['animation.ffmpeg_path'] = ffmpeg_exec
                except Exception:
                    pass
                Writer = animation.FFMpegWriter
                writer = Writer(fps=fps or int(round(1.0 / (bvh.frame_time * step))) if bvh.frame_time > 0 else 30, bitrate=2400)
                ani.save(out_path, writer=writer, dpi=dpi)
            elif ext in (".gif",):
                try:
                    from matplotlib.animation import PillowWriter  # type: ignore
                    writer = PillowWriter(fps=fps or int(round(1.0 / (bvh.frame_time * step))) if bvh.frame_time > 0 else 15)
                    ani.save(out_path, writer=writer, dpi=dpi)
                except Exception as e:
                    print(f"Failed to use PillowWriter for GIF: {e}")
                    raise
            else:
                # default to mp4
                ffmpeg_exec = shutil.which("ffmpeg")
                if ffmpeg_exec is None:
                    try:
                        import imageio_ffmpeg  # type: ignore
                        ffmpeg_exec = imageio_ffmpeg.get_ffmpeg_exe()
                    except Exception:
                        ffmpeg_exec = None
                if ffmpeg_exec is None:
                    raise RuntimeError(
                        "ffmpeg not found. Install via 'sudo apt-get install ffmpeg' or 'conda install -c conda-forge ffmpeg', "
                        "or save as GIF with --save output.gif (requires pillow)."
                    )
                try:
                    from matplotlib import rcParams
                    rcParams['animation.ffmpeg_path'] = ffmpeg_exec
                except Exception:
                    pass
                Writer = animation.FFMpegWriter
                writer = Writer(fps=fps or int(round(1.0 / (bvh.frame_time * step))) if bvh.frame_time > 0 else 30, bitrate=2400)
                ani.save(out_path, writer=writer, dpi=dpi)
            print(f"Saved animation to: {out_path}")
        finally:
            plt.close(fig)
    else:
        plt.show()


# -------------------------------
# CLI
# -------------------------------

def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Read and visualize CMU mocap BVH files")
    parser.add_argument("bvh", nargs='?', default=None, help="Path to a .bvh file. If omitted, the first file found will be used.")
    parser.add_argument("--root", default="/data/yanyu/GF-theory/Mocap/cmu-mocap/data", help="Root directory to search for BVH files")
    parser.add_argument("--list", action="store_true", help="List available BVH files and exit")
    parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame for animation")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to visualize")
    parser.add_argument("--save", default=None, help="Path to save animation (e.g., output.mp4 or output.gif)")
    parser.add_argument("--fps", type=int, default=None, help="FPS for saved animation (overrides BVH frame rate)")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for saved animation frames")
    parser.add_argument("--interactive", action="store_true", help="Open interactive viewer with slider and keyboard controls")
    parser.add_argument("--backend", default=None, help="Matplotlib backend to use (e.g., TkAgg, Qt5Agg, MacOSX). For headless, skip interactive.")
    parser.add_argument("--export_frames", default=None, help="Directory to export per-frame PNGs (frame_000000.png, ...)")

    args = parser.parse_args(argv)

    if args.list:
        files = list_bvh_files(args.root)
        for p in files:
            print(p)
        return 0

    target = args.bvh
    if target is None:
        files = list_bvh_files(args.root)
        if not files:
            print(f"No BVH files found under {args.root}", file=sys.stderr)
            return 1
        target = files[0]

    # Backend control and headless notice
    if args.backend and plt is not None:
        try:
            plt.switch_backend(args.backend)
        except Exception as e:
            print(f"Warning: failed to switch backend to {args.backend}: {e}")

    if args.interactive and plt is not None:
        try:
            mgr = plt.get_backend()
            # Common GUI backends
            gui_ok = any(x in str(mgr) for x in ["TkAgg", "Qt", "Qt5", "QtAgg", "MacOSX", "WXAgg"])
            if not gui_ok:
                print("Notice: current matplotlib backend may be non-GUI (headless). The interactive window might not appear. Try --backend TkAgg or Qt5Agg, or use --save.")
        except Exception:
            pass

    visualize_bvh(target, downsample=args.downsample, max_frames=args.max_frames, save=args.save, fps=args.fps, dpi=args.dpi, interactive=args.interactive, export_frames=args.export_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
