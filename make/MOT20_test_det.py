import os
import cv2

# 读取det.txt文件
with open(r'/dataset/MOT20/test/MOT20-08/det/det.txt', 'r') as file:
    data = file.readlines()

# 指定保存剪切出来的图片的文件夹
output_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset\output_image\MOT20_test'
input_folder = r'E:\work\SR-YOLO\dataset\MOT20\test\MOT20-08\img1'

# 遍历det.txt中的每一行数据
for line in data:
    info = line.split(',')
    frame_num = int(info[0])
    x = int(info[2])
    y = int(info[3])
    width = int(info[4])
    height = int(info[5])

    # 读取对应帧的图片
    # image_path = 'path_to_dataset_folder/{:06d}.jpg'.format(frame_num)
    image_path = os.path.join(input_folder, '{:06d}.jpg'.format(frame_num))
    image = cv2.imread(image_path)

    # 剪切出行人的图片
    person_image = image[y:y+height, x:x+width]

    # 保存剪切出来的图片到指定文件夹
    output_path = os.path.join(output_folder, '{:06d}.jpg'.format(frame_num))
    cv2.imwrite(output_path, person_image)

print("剪切并保存图片完成")