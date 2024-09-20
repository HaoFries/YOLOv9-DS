import os
import random

# 数据集划分比例
trainval_percent = 0.9  # 用于训练和验证的数据比例
train_percent = 0.9     # 训练集在 trainval 中的比例

# YOLO标签文件夹路径
labels_path = r'E:\work\YOLO\yolov9-main\datasets\labels'  # 这里修改为YOLO格式的标签路径
txtsavepath = r'E:\work\YOLO\yolov9-main\datasets\ImageSets'

# 获取所有标签文件
total_labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

num = len(total_labels)
list_indices = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)

# 随机抽样划分
trainval = random.sample(list_indices, tv)
train = random.sample(trainval, tr)

# 确保保存路径存在
os.makedirs(txtsavepath, exist_ok=True)

# 打开输出文件
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

# 写入数据集划分的文件名（不含扩展名）
for i in list_indices:
    name = total_labels[i][:-4] + '\n'  # 去掉 .txt 扩展名
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

# 关闭所有文件
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print(f"Dataset split completed: {tr} train, {tv-tr} val, {num-tv} test images.")