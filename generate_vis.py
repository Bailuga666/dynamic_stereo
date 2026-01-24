import cv2
import numpy as np
import argparse
import os
import glob

def load_sequence_npy(path: str) -> np.ndarray:
    data = np.load(path)
    # Handle various potential shapes (H,W), (1,H,W), (T,H,W)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    return data.astype(np.float32)

def disparity_to_depth(disp: np.ndarray, scale=None, eps=1e-6) -> np.ndarray:
    disp_abs = np.abs(disp)
    if scale is None:
        depth = 1.0 / (disp_abs + eps)
    else:
        depth = scale / (disp_abs + eps)
    
    # Mask out inf/nan
    depth[np.logical_not(np.isfinite(depth))] = 0
    return depth

def normalize_u8(depth: np.ndarray, lower_pct=1.0, upper_pct=99.0) -> np.ndarray:
    # Normalize per frame or global? The sample uses frame-wise normalization
    valid_mask = (depth > 0)
    if not np.any(valid_mask):
         return np.zeros_like(depth, dtype=np.uint8)
         
    vals = depth[valid_mask]
    vmin = np.percentile(vals, lower_pct)
    vmax = np.percentile(vals, upper_pct)
    
    if vmax <= vmin:
        return np.zeros_like(depth, dtype=np.uint8)
        
    depth_clipped = np.clip(depth, vmin, vmax)
    norm = (depth_clipped - vmin) / (vmax - vmin)
    return (np.clip(norm * 255, 0, 255)).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.disp_dir, "*.npy")))
    if not files:
        print("No .npy files found")
        return

    cmap_dict = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "viridis_r": cv2.COLORMAP_VIRIDIS # Handle inversion manually
    }
    cmap_code = cmap_dict.get(args.cmap, cv2.COLORMAP_MAGMA)
    
    images = []
    
    print(f"Generating images in {args.out_dir}...")
    
    H, W = 0, 0
    
    for f in files:
        basename = os.path.basename(f)
        name_no_ext = os.path.splitext(basename)[0]
        
        disp = load_sequence_npy(f)
        depth = disparity_to_depth(disp)

        depth_u8 = normalize_u8(depth)
        
        if args.cmap == "viridis_r":
            depth_u8 = 255 - depth_u8
            
        color = cv2.applyColorMap(depth_u8, cmap_code)
        
        out_path = os.path.join(args.out_dir, name_no_ext + ".png")
        cv2.imwrite(out_path, color)
        
        images.append(color)
        if H == 0:
            H, W, _ = color.shape

    # Create video
    video_path = os.path.join(args.out_dir, "depth_video.mp4")
    if images:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, args.fps, (W, H))
        for img in images:
            writer.write(img)
        writer.release()
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()
