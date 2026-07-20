import os
import tempfile
from pathlib import Path
import argparse
import rasterio
from rasterio.merge import merge
import numpy as np

def assign_coords_from_a(b_path, a_path, out_path):
    """
    将A的地理坐标（transform + crs）赋给B，生成带坐标的tif
    """
    with rasterio.open(a_path) as src_a:
        profile = src_a.profile.copy()
        with rasterio.open(b_path) as src_b:
            # 更新profile：保留A的坐标，但用B的尺寸、波段数、数据类型
            profile.update({
                'height': src_b.height,
                'width': src_b.width,
                'count': src_b.count,
                'dtype': src_b.dtypes[0],   # 假设所有波段类型一致
                'compress': 'lzw',           # 可选压缩
            })
            data = src_b.read()   # 读取所有波段 (count, h, w)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data)

def main():
    parser = argparse.ArgumentParser(
        description='为无坐标tif块(B)赋予同名tif(A)的坐标，按文件名顺序合并为一张带坐标的大图'
    )
    parser.add_argument('--b_dir', default=r'E:\MM-DINO-main\MM-DINO-main\tasks\segmentation\vis_results\csgo2_predict_totalRGB', help='B文件夹，存放无坐标的tif')
    parser.add_argument('--a_dir', default=r'E:\segdino-main\segdino-main\segdata\csgo2\test\image', help='A文件夹，存放有坐标的tif（文件名需与B同名）')
    parser.add_argument('--output', default=r'E:\result_csgo\BIG_IMAGE\csgo2.tif', help='输出大图路径（.tif）')
    parser.add_argument('--temp_dir', help='临时目录（可选，默认系统临时目录）')
    args = parser.parse_args()

    b_dir = Path(args.b_dir)
    a_dir = Path(args.a_dir)
    out_path = Path(args.output)

    # 按文件名排序获取B文件夹中所有tif
    b_files = sorted(b_dir.glob('*.tif'))
    if not b_files:
        print("错误：B文件夹中没有 .tif 文件")
        return

    # 创建临时目录存放赋坐标后的临时tif
    with tempfile.TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        geo_files = []

        for b_file in b_files:
            a_file = a_dir / b_file.name
            if not a_file.exists():
                print(f"警告：A文件夹中找不到 {b_file.name}，跳过")
                continue

            geo_path = os.path.join(tmpdir, b_file.name)
            assign_coords_from_a(str(b_file), str(a_file), geo_path)
            geo_files.append(geo_path)
            print(f"✓ 已处理: {b_file.name}")

        if not geo_files:
            print("错误：没有成功处理的文件，无法合并")
            return

        # 合并所有带坐标的tif块
        print("正在合并所有块...")
        srcs = [rasterio.open(f) for f in geo_files]
        mosaic, out_transform = merge(srcs)
        for src in srcs:
            src.close()

        # 获取合并后的元数据（投影从第一个源文件继承）
        with rasterio.open(geo_files[0]) as first:
            out_meta = first.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'compress': 'lzw',
        })

        # 写入最终大图
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(mosaic)

        print(f"✅ 合并完成，输出至: {out_path}")

if __name__ == '__main__':
    main()