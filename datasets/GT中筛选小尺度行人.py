
def filter_small_scale_pedestrians(input_file, output_file, width_threshold=50, height_threshold=50):
    total_objects = 0
    small_objects = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = line.strip().split(',')
            frame_id = data[0]
            identity_id = data[1]
            x_min = int(data[2])
            y_min = int(data[3])
            width = int(data[4])
            height = int(data[5])
            flag = int(data[6])
            category = int(data[7])
            visibility = float(data[8])

            # 统计总目标数量
            total_objects += 1

            # 判断是否符合条件
            if flag == 1 and category == 1 and width < width_threshold and height < height_threshold:
                small_objects += 1
                outfile.write(
                    f"{frame_id},{identity_id},{x_min},{y_min},{width},{height},{flag},{category},{visibility}\n")

    # 计算小目标占比
    small_objects_ratio = small_objects / total_objects if total_objects else 0

    # 输出统计信息
    print(f"Total objects: {total_objects}")
    print(f"Small objects: {small_objects}")
    print(f"Small objects ratio: {small_objects_ratio:.2%}")

# 使用示例
input_file = ''  # 输入GT文件路径
output_file = ''  # 输出文件路径

filter_small_scale_pedestrians(input_file, output_file)