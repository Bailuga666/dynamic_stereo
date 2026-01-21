import os
import glob
import numpy as np
import cv2
import argparse

DISPARITY_DIR = "/openbayes/home/dynamic_stereo/outputs/custom_eval/disparities"
OUTPUT_VIDEO = "/openbayes/home/dynamic_stereo/outputs/custom_eval/depth_video.mp4"
FPS = 10
EPS = 1e-6


def normalize_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - vmin) / (vmax - vmin)
    return (np.clip(norm * 255, 0, 255)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disp_dir", default=DISPARITY_DIR, help="Disparity .npy directory")
    parser.add_argument("--out", default=OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--fps", type=int, default=FPS, help="Output FPS")
    parser.add_argument("--eps", type=float, default=EPS, help="Small epsilon for depth conversion")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.disp_dir, "*.npy")))
    if not files:
        print("没有找到视差文件")
        return

    # 读取所有视差帧，合成序列 (T,H,W)
    disps = []
    for f in files:
        d = np.load(f)
        if d.ndim == 2:
            disps.append(d.astype(np.float32))
        elif d.ndim == 3 and d.shape[0] == 1:
            disps.append(d[0].astype(np.float32))
        else:
            raise RuntimeError(f"Unsupported disparity shape in {f}: {d.shape}")

    seq = np.stack(disps, axis=0)  # (T,H,W)
    T, H, W = seq.shape

    # 将视差转为深度（全序列统一归一化）
    disp_abs = np.abs(seq)
    depth = 1.0 / (disp_abs + args.eps)

    # 屏蔽掉无穷/NaN
    depth[np.logical_or(np.isnan(depth), np.isinf(depth))] = 0.0

    vmin = float(np.min(depth[depth > 0])) if np.any(depth > 0) else 0.0
    vmax = float(np.max(depth)) if np.any(depth > 0) else 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    for i in range(T):
        depth_frame = depth[i]
        depth_u8 = normalize_to_uint8(depth_frame, vmin, vmax)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        writer.write(depth_color)

    writer.release()
    print(f"已生成深度视频: {args.out} (frames={T}, size={W}x{H})")


if __name__ == "__main__":
    main()
