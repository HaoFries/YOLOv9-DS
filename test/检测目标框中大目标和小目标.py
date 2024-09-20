def analyze_boxes(file_path):
    small_count = 0  # 宽度和高度不大于32的计数
    large_count = 0  # 宽度和高度大于32的计数

    # 读取并分析文件
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                data = line.strip().split(',')
                # 检查第5列和第6列的值（索引为4和5）
                width = float(data[4])
                height = float(data[5])
                # 根据宽度和高度分类计数
                if width <= 32 and height <= 32:
                    small_count += 1
                else:
                    large_count += 1

    return small_count, large_count

# 输入文件路径
input_file = r'E:\work\small_person_AP\gt_Person_X0.25.txt'  # 替换为实际文件路径

# 分析文件并获取结果
small_boxes, large_boxes = analyze_boxes(input_file)

# 在终端输出结果
print(f"目标框宽和高不大于32的个数: {small_boxes}")
print(f"目标框宽和高大于32的个数: {large_boxes}")