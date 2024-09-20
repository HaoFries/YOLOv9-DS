#有效保留图片中移动的行人和静止的行人的信息。
import os

def multiply_columns_and_filter(input_file, output_dir, scale=0.125):
    # 获取输入文件的文件名和扩展名
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)

    # 构造输出文件路径，使用新的后缀
    output_file = os.path.join(output_dir, f"{name}_Person_X{scale}.txt")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取输入文件并处理
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():  # 确保不处理空行
                data = line.strip().split(',')
                # 检查第7列和第8列的值是否为"1,1"或者"1,7"
                if (data[6] == '1' and data[7] == '1') or (data[6] == '0' and data[7] == '7') or (data[6] == '1' and data[7] == '7'):
                    # 将第3, 4, 5, 6列（索引为2, 3, 4, 5）的数字乘以scale
                    for i in [2, 3, 4, 5]:  # 使用列表提高可读性
                        data[i] = str(float(data[i]) * scale)
                    # 重新写入文件
                    outfile.write(','.join(data) + '\n')

    print(f"处理后的文件已保存为 {output_file}")

# 使用示例
input_file = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt\gt.txt'  # 输入文件路径
output_dir = r'E:\work\small_person_AP'  # 输出文件夹路径

# 调用函数，数值缩小4倍并根据条件过滤
multiply_columns_and_filter(input_file, output_dir)