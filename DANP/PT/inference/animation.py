import cv2
import os

# 图片目录路径
image_dir = os.path.expanduser("../inference/output/chicago")

# 输出视频文件路径
output_video = os.path.expanduser("../inference/output/chicago/video.mp4")

# 获取图片目录下的所有文件名，并按文件名排序
image_files = sorted(os.listdir(image_dir))

# 获取第一张图片的尺寸作为视频的尺寸
first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width, _ = first_image.shape

# 创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

# 逐帧写入视频
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(image_file)
    frame = cv2.imread(image_path)
    video_writer.write(frame)

# 释放资源
video_writer.release()