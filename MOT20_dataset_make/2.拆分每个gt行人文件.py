#############直接得到gt转为YOLO格式的每一帧的gt文件##########
from pathlib import Path
from collections import defaultdict

def split_and_modify_gt_file_by_frame(gt_file_path, output_dir, img_width, img_height):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 defaultdict 将帧序号与对应行信息存储在一起
    frame_data = defaultdict(list)

    # 读取 gt 文件并将数据存储在字典中
    with open(gt_file_path, 'r') as gt_file:
        for line in gt_file:
            parts = line.strip().split(',')
            frame_id = parts[0]  # 获取帧序号

            # 获取原始的 x, y, w, h
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # 计算中心点坐标
            cx = x + w / 2
            cy = y + h / 2

            # 归一化中心点坐标和宽高
            cx_norm = cx / img_width
            cy_norm = cy / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 修改第一列为整数0，并将归一化后的值写入
            modified_parts = f'0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}'

            # 将修改后的行信息存储在一起
            frame_data[frame_id].append(modified_parts)

    # 将每个帧序号的信息写入单独的 txt 文件
    for frame_id, lines in frame_data.items():
        output_file_path = output_dir / f"{frame_id}.txt"
        with open(output_file_path, 'w') as output_file:
            output_file.write('\n'.join(lines))

    print(f"GT 文件已按照帧序号分割、修改并保存在 {output_dir} 文件夹中")

# 自定义的图像宽度和高度
img_width = 1920  # 根据你的需要调整宽度
img_height = 1080  # 根据你的需要调整高度

# 文件路径设置
gt_file_path = ""  # gt.txt 文件路径
output_dir = ""  # 输出的 txt 文件保存的目录

# 按帧序号分割、修改并保存 gt 文件
split_and_modify_gt_file_by_frame(gt_file_path, output_dir, img_width, img_height)