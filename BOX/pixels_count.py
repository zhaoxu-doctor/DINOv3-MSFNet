import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

def count_pixels_in_folder(folder_path: str) -> dict:
    """
    遍历文件夹内的所有 .tif 文件，累加每个文件里像素值为1、2、3的数量。
    返回 {'1': 总数, '2': 总数, '3': 总数}。
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不存在或不是文件夹: {folder_path}")

    counts = {1: 0, 2: 0, 3: 0}
    tif_files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.TIF"))
    if not tif_files:
        print("警告：文件夹内没有找到 .tif 文件")
        return counts

    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                # 读取第一波段（2D数组）
                data = src.read(1).astype(np.uint8)  # 确保类型一致
        except RasterioIOError as e:
            print(f"无法读取 {tif_path.name}，跳过：{e}")
            continue

        # 累加各值的数量
        counts[1] += int(np.sum(data == 1))
        counts[2] += int(np.sum(data == 2))
        counts[3] += int(np.sum(data == 3))

    return counts

def main():
    parser = argparse.ArgumentParser(
        description="统计文件夹内所有tif影像中像素值为1、2、3的总数"
    )
    parser.add_argument("--folder",default=r'E:\segdino-main\segdino-main\segdata\csgo\test\mask', help="存放tif影像的文件夹路径")
    args = parser.parse_args()

    try:
        result = count_pixels_in_folder(args.folder)
    except Exception as e:
        print(f"错误：{e}")
        return

    print("统计结果（所有tif文件合计）：")
    print(f"  像素值 1 : {result[1]}")
    print(f"  像素值 2 : {result[2]}")
    print(f"  像素值 3 : {result[3]}")

if __name__ == "__main__":
    main()