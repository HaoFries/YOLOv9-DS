#######生成MOT20数据集的新的只保留行人和伫立的人的gt文件在gt1235中保存
from pathlib import Path


def process_gt_file(gt_file_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_path = output_dir / "gt3.txt"  # 调整gt文件名称编号

    with open(gt_file_path, 'r') as gt_file, open(output_file_path, 'w') as output_file:
        lines = gt_file.readlines()

        for line in lines:
            parts = line.strip().split(',')

            if len(parts) < 8:
                continue  # 确保有足够的列

            col_7 = int(parts[6])  # 第7列
            col_8 = int(parts[7])  # 第8列

            if col_7 == 1 and col_8 in [1, 7]:
                # 保留符合条件的行
                output_file.write(line)

    print(f"Processed gt file saved to {output_file_path}")


# 文件路径设置
gt_file_path = ""  # gt3.txt 文件路径
output_dir = ""  # 重新生成的文件保存的路径

# 处理 gt 文件
process_gt_file(gt_file_path, output_dir)