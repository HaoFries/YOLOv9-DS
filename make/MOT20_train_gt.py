import cv2
import os

# 读取gt.txt文件
with open(r'/dataset/MOT20/train/MOT20-03/gt/gt.txt', 'r') as file:
    data = file.readlines()

# 指定文件夹路径
input_folder = r'E:\work\SR-YOLO\dataset\MOT20\train\MOT20-03\img1'
output_parent_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset_make\output_image\cut_person\MOT20-03-HR'

# 初始化一个字典，用于存储每个物体编号对应的文件夹路径
object_folders = {}

# 循环处理每一行数据
for line in data:
    values = line.split(',')
    frame_number = int(values[0])
    object_id = int(values[1])
    x, y, w, h = map(int, values[2:6])
    category_id = int(values[7])

    # 只处理类别为1的行人
    if category_id == 1:
        # 读取对应帧的图像
        image_path = os.path.join(input_folder, '{:06d}.jpg'.format(frame_number))
        image = cv2.imread(image_path)

        # 剪切行人并保存到对应物体编号的文件夹
        if image is not None:
            person_image = image[y:y + h, x:x + w]
            # 检查物体编号对应的文件夹是否已经存在，如果不存在则创建
            if object_id not in object_folders:
                object_folder_path = os.path.join(output_parent_folder, 'person_{}'.format(object_id))
                os.makedirs(object_folder_path, exist_ok=True)
                object_folders[object_id] = object_folder_path
            output_path = os.path.join(object_folders[object_id], '{:06d}.jpg'.format(frame_number))
            cv2.imwrite(output_path, person_image)

print("处理完成")

#====================没有分组，直接生成=====================
# import cv2
# import os
#
# # 读取gt.txt文件
# with open(r'E:\work\SR-YOLO\dataset\MOT20\train\MOT20-01\gt\gt.txt', 'r') as file:
#     data = file.readlines()
#
# # 指定文件夹路径
# input_folder = r'E:\work\SR-YOLO\dataset\MOT20\train\MOT20-01\img1'
# output_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset\output_image'
#
# # 循环处理每一行数据
# for line in data:
#     values = line.split(',')
#     frame_number = int(values[0])
#     x, y, w, h = map(int, values[2:6])
#     category_id = int(values[7])
#
#     # 只处理类别为1的行人
#     if category_id == 1:
#         # 读取对应帧的图像
#         image_path = os.path.join(input_folder, '{:06d}.jpg'.format(frame_number))
#         image = cv2.imread(image_path)
#
#         # 剪切行人并保存到指定文件夹
#         if image is not None:
#             person_image = image[y:y + h, x:x + w]
#             output_path = os.path.join(output_folder, 'person_{:06d}.jpg'.format(frame_number))
#             cv2.imwrite(output_path, person_image)
#
# print("处理完成")