import os
import cv2
import numpy as np

# 输入视频
VIDEO_PATH = "/openbayes/home/bidastereo/your_mono_video.mp4"

# 输出目录
OUTPUT_DIR = "/openbayes/home/dynamic_stereo/my_data"
LEFT_DIR = os.path.join(OUTPUT_DIR, "left")
RIGHT_DIR = os.path.join(OUTPUT_DIR, "right")

# 处理参数
CROP_SIZE = 256
SHIFT_PIXELS = 20  # 右视图相对左视图向右平移的像素数
MAX_FRAMES = 100


def main():
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = CROP_SIZE // 2

        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

        left_path = os.path.join(LEFT_DIR, f"{frame_count:04d}.png")
        cv2.imwrite(left_path, crop)

        M = np.float32([[1, 0, SHIFT_PIXELS], [0, 1, 0]])
        right = cv2.warpAffine(crop, M, (CROP_SIZE, CROP_SIZE), borderMode=cv2.BORDER_REPLICATE)
        right_path = os.path.join(RIGHT_DIR, f"{frame_count:04d}.png")
        cv2.imwrite(right_path, right)

        frame_count += 1
        if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
            break

    cap.release()
    print(f"已处理 {frame_count} 帧，输出到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
