import os
import numpy as np
from PIL import Image


def convert_label_images(mode, input_folder, output_folder=None):
    """
    将 input_folder 中所有像素值为 0~3 的标签图像转换为彩色图像。
    颜色映射：
        0 -> #FFFFFF (白色)
        1 -> #172523 (深绿)
        2 -> #00FE01 (亮绿)
        3 -> #FFFF00 (黄色)

    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str, optional): 输出文件夹路径。若为 None，则自动创建以
                                       “原文件夹名_converted” 命名的文件夹
    """
    # 颜色查找表 (4 种颜色, 每种 RGB)
    if mode == 'csgo':
        lut = np.array([
            [255, 255, 255],   # 0: 白色
            [  0, 90,   0],   # 1: #172523
            [  0, 254,   1],   # 2: #00FE01
            [255, 255,   0]    # 3: #FFFF00
        ], dtype=np.uint8)
    else:
        lut = np.array([
            [255, 255, 255],  # 0: 裸地退化 - 浅灰白裸土
            [  0, 254,   1],  # 1: 草原 - 浅草绿
            [255, 255,   0],  # 2: 异质农业区 - 土黄农田
            [0, 90, 0],  # 3: 茂密森林 - 深墨绿
            [30, 110, 220],  # 4: 水体 - 标准蓝色
            [160, 40, 40]  # 5: 建成区 - 砖红建筑
        ], dtype=np.uint8)

    # 确定输出文件夹
    if output_folder is None:
        parent = os.path.dirname(input_folder) or '.'
        basename = os.path.basename(input_folder) or 'labels'
        output_folder = os.path.join(parent, basename + '_converted')
    os.makedirs(output_folder, exist_ok=True)

    # 支持的文件扩展名
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.tiff')

    # 获取所有图像文件并排序
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith(valid_ext)]
    files.sort()

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        img = Image.open(filepath)

        # 确保图像为灰度（单通道）
        if img.mode != 'L':
            img = img.convert('L')
        arr = np.array(img, dtype=np.int32)

        # ---- 安全处理：将像素值限定在 0~3 范围 ----
        # （如果图像中有其他值，会将其映射到白色，可根据需要调整）
        if mode == 'csgo':
            arr = np.clip(arr, 0, 3)
        else:
            arr = np.clip(arr, 0, 5)

        # ---- 向量化颜色映射 ----
        # 利用 numpy 高级索引: lut[arr] 会得到一个 (H, W, 3) 数组
        colored = lut[arr]

        # 保存为彩色图像（自动匹配原文件名和扩展名）
        out_img = Image.fromarray(colored, mode='RGB')
        out_path = os.path.join(output_folder, filename)
        out_img.save(out_path)
        print(f'✔ 已转换: {filename}')

    print(f'\n✅ 全部处理完成，输出文件夹: {output_folder}')


# ===== 使用示例 =====
if __name__ == '__main__':
    # 请修改为你的标签图像所在文件夹路径
    mode = 'TALANDCOVER'
    INPUT_DIR = r'E:\segdino-main\segdino-main\segdata\TALANDCOVER\test\labels'
    OUTPUT_DIR = r'E:\segdino-main\segdino-main\segdata\TALANDCOVER\test\labels_RGB'
    convert_label_images(mode, INPUT_DIR, OUTPUT_DIR)