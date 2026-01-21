import os
import glob
import numpy as np
import cv2
from flow_vis import flow_to_color

DISPARITY_DIR = "/openbayes/home/dynamic_stereo/outputs/custom_eval/disparities"
OUTPUT_VIDEO = "/openbayes/home/dynamic_stereo/outputs/custom_eval/depth_video.mp4"
FPS = 10


def main():
    files = sorted(glob.glob(os.path.join(DISPARITY_DIR, "*.npy")))
    if not files:
        print("没有找到视差文件")
        return

    first = np.load(files[0])
    h, w = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    for f in files:
        disp = np.load(f)
        # 归一化视差到0-1，用于flow_vis
        disp_min = disp.min()
        disp_max = disp.max()
        if disp_max > disp_min:
            disp_norm = (disp - disp_min) / (disp_max - disp_min)
        else:
            disp_norm = np.zeros_like(disp)
        
        # 将视差当作光流可视化（x=0, y=disp）
        flow = np.zeros((h, w, 2))
        flow[:, :, 1] = disp_norm * 10  # 放大以便可视化
        
        # 转换为彩色图像
        color_img = flow_to_color(flow)
        writer.write(color_img)

    writer.release()
    print(f"已生成深度视频: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
