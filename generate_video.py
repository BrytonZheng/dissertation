import cv2
import os
import imageio

# 设置参数
image_folder = 'pic_new/1-1'  # 图片所在的文件夹
output_video = 'pic_output2-1.mp4'  # 输出视频文件名
fps = 5  # 设定视频帧率

# 获取所有图片文件
images = sorted(
    [int(img[0:-4]) for img in [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]])

# 读取第一张图片以获取尺寸信息
first_image = cv2.imread(os.path.join(image_folder, str(images[0]) + '.png'))
height, width, _ = first_image.shape

# 使用 imageio 生成视频
video_writer = imageio.get_writer(output_video, fps = fps)

for image in images:
    img_path = os.path.join(image_folder, str(image) + '.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 读取的图像是 BGR 需要转换为 RGB
    video_writer.append_data(img)

video_writer.close()

print(f"视频已保存为 {output_video}")
