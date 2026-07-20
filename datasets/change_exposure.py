import os
from osgeo import gdal
import numpy as np


def enhance_brightness(input_folder, output_folder, brightness_factor=1.3):
    """
    读取文件夹内的遥感影像，提高亮度，保留地理坐标

    参数:
        input_folder: 输入影像文件夹路径
        output_folder: 输出影像文件夹路径
        brightness_factor: 亮度增强系数，默认1.3（增加30%亮度）
    """

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的影像格式
    supported_formats = ['.tif', '.tiff', '.img', '.jpg', '.jpeg', '.png']

    # 遍历输入文件夹
    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in supported_formats:
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"处理: {filename}")

        try:
            # 打开影像
            dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"  无法打开: {filename}")
                continue

            # 获取影像信息
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            bands = dataset.RasterCount

            # 获取地理坐标信息
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()

            # 创建输出影像
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(
                output_path,
                cols,
                rows,
                bands,
                dataset.GetRasterBand(1).DataType
            )

            # 设置地理坐标
            out_dataset.SetGeoTransform(geotransform)
            out_dataset.SetProjection(projection)

            # 逐波段处理
            for i in range(1, bands + 1):
                band = dataset.GetRasterBand(i)
                data = band.ReadAsArray()

                # 获取数据类型信息
                data_type = band.DataType
                nodata = band.GetNoDataValue()

                # 提高亮度
                if nodata is not None:
                    # 创建掩膜，排除无效值
                    mask = data != nodata
                    enhanced = data.astype(np.float32)
                    enhanced[mask] = enhanced[mask] * brightness_factor
                else:
                    enhanced = data.astype(np.float32) * brightness_factor

                # 限制数值范围，避免溢出
                if data_type == gdal.GDT_Byte:
                    enhanced = np.clip(enhanced, 0, 255)
                elif data_type == gdal.GDT_UInt16:
                    enhanced = np.clip(enhanced, 0, 65535)

                # 转换回原始数据类型
                enhanced = enhanced.astype(data.dtype)

                # 写入输出影像
                out_band = out_dataset.GetRasterBand(i)
                out_band.WriteArray(enhanced)

                if nodata is not None:
                    out_band.SetNoDataValue(nodata)

                # 复制颜色表（如果有）
                color_table = band.GetColorTable()
                if color_table:
                    out_band.SetColorTable(color_table)

            # 关闭数据集
            dataset = None
            out_dataset = None

            print(f"  完成: {filename}")

        except Exception as e:
            print(f"  处理失败 {filename}: {str(e)}")
            continue


# 使用示例
if __name__ == "__main__":
    input_folder = r"E:\segdino-main\segdino-main\segdata\csgo2\test0\image"  # 输入文件夹路径
    output_folder = r"E:\segdino-main\segdino-main\segdata\csgo2\test\image"  # 输出文件夹路径
    brightness_factor = 2  # 亮度增强系数

    enhance_brightness(input_folder, output_folder, brightness_factor)
    print("所有影像处理完成！")