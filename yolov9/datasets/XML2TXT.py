import os
from os import getcwd
from os.path import join

# 定义数据集类型
sets = ['train', 'test', 'val']

# 返回当前工作目录
wd = getcwd()
print(f"Current working directory: {wd}")

# 检查并创建 labels 目录
if not os.path.exists('E:/work/YOLO/yolov9-main/datasets/labels/'):
    os.makedirs('E:/work/YOLO/yolov9-main/datasets/labels/')

for image_set in sets:
    '''
    对所有的数据集（train、test、val）进行处理
    生成对应的 train.txt, test.txt, val.txt 文件，其中包含图像的路径。
    '''
    # 读取 ImageSets 文件夹中对应的数据集文件
    image_ids = open(f'E:/work/YOLO/yolov9-main/datasets/ImageSets/{image_set}.txt').read().strip().split()

    # 打开对应的数据集路径文件进行写入
    list_file_path = f'E:/work/YOLO/yolov9-main/datasets/{image_set}.txt'
    list_file = open(list_file_path, 'w')

    # 遍历所有图片 ID，并将图片路径写入列表文件
    for image_id in image_ids:
        # 构建图像的完整路径
        image_path = f'E:/work/YOLO/yolov9-main/datasets/images/{image_id}.jpg'

        # 检查图像文件是否存在
        if os.path.exists(image_path):
            list_file.write(image_path + '\n')
        else:
            print(f"Warning: {image_path} does not exist.")

    # 关闭文件
    list_file.close()
    print(f"Image list file generated: {list_file_path}")