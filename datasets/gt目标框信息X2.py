import os


def multiply_columns(input_file, output_dir):
    # 获取输入文件的文件名和扩展名
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)

    # 构造输出文件路径
    output_file = os.path.join(output_dir, f"{name}_X2{ext}")

    # 读取输入文件并处理
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = line.strip().split(',')
            # 将第3, 4, 5, 6列（索引为2, 3, 4, 5）的数字乘以2
            for i in range(2, 6):
                data[i] = str(int(data[i]) * 2)
            # 重新写入文件
            outfile.write(','.join(data) + '\n')

    print(f"处理后的文件已保存为 {output_file}")


# 使用示例
input_file = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt\gt_small_scale_pedestrians.txt'  # 输入文件路径
output_dir = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt'  # 输出文件夹路径，已自动添加_X2后缀。

multiply_columns(input_file, output_dir)