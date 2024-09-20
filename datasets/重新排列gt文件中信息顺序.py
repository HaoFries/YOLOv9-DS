# 定义函数，用于读取文件、排序并写入新文件
def sort_file(input_file, output_file):
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 将每一行的内容分割并转换为列表，存储为元组列表
    data = [tuple(map(float, line.strip().split(','))) for line in lines]

    # 按照第一列进行排序
    sorted_data = sorted(data, key=lambda x: x[0])

    # 将排序后的数据转换为字符串，并将除最后一列外的列转换为整数
    sorted_lines = []
    for row in sorted_data:
        int_part = map(int, row[:-1])  # 除最后一列外的列转换为整数
        float_part = [row[-1]]  # 最后一列保持浮点数
        sorted_lines.append(','.join(map(str, list(int_part) + float_part)))

    # 将排序后的内容写入新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sorted_lines:
            f.write(line + '\n')

# 主程序
if __name__ == '__main__':
    # 输入文件名和输出文件名
    input_file = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt\gt_small_scale_pedestrians.txt'  # 指定你的输入文件名
    output_file = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt\gt_small_scale_pedestrians-1-481.txt'  # 指定你的输出文件名

    # 调用函数进行排序
    sort_file(input_file, output_file)

    print(f'文件 {input_file} 的内容已按第一列排序并写入文件 {output_file}。')