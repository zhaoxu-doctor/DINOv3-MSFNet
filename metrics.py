import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

matplotlib.rcParams['axes.unicode_minus'] = False  # 处理坐标轴负号显示问题

def metrics_print_version(predictions, gts, label_values):
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score / IoU:")
    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                          np.diag(cm))
    for l_id, (f1_score, iou_score) in enumerate(zip(F1Score, MIoU)):
        print("%s: %.2f / %.2f" %
              (label_values[l_id], f1_score * 100, iou_score * 100))
    if "undefined" in label_values or "clutter" in label_values:
        F1Score = np.nanmean(F1Score[:(len(label_values) - 1)])
    else:
        F1Score = np.nanmean(F1Score[:(len(label_values))])
    print("---")
    print('mean F1Score: %.4f' % (F1Score))
    print("---")
    if "undefined" in label_values or "clutter" in label_values:
        MIoU = np.nanmean(MIoU[:(len(label_values) - 1)])
    else:
        MIoU = np.nanmean(MIoU[:(len(label_values))])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))

    return MIoU, F1Score, kappa, accuracy, cm
def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵图像"""
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import os

    # 尝试设置支持中文的字体
    try:
        # 定义常见的中文字体路径
        zh_font_paths = [
            # Noto Sans CJK 字体路径
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
            # 文泉驿字体路径
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            # 其他可能的路径
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
        ]

        # 首先尝试直接指定的字体路径
        font_found = False
        for font_path in zh_font_paths:
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                font_found = True
                break

        # 如果直接路径找不到，再尝试搜索字体名称
        if not font_found:
            # 搜索包含中文字符的字体
            zh_fonts = []
            for font in font_manager.fontManager.ttflist:
                font_name_lower = font.name.lower()
                # 检查是否包含中文字体关键词
                if any(keyword in font_name_lower for keyword in [
                        'noto', 'cjk', 'wqy', 'micro', 'hei', 'song', 'kai',
                        'sans', 'serif'
                ]):
                    # 额外验证字体是否支持中文字符（通过检查family name）
                    if any(chinese_indicator in font.name
                           for chinese_indicator in
                           ['SC', 'TC', 'JP', 'KR', 'HK', 'CJK', 'CHINESE']):
                        zh_fonts.append(font)
                    # 或者检查文件名是否包含中文指示符
                    elif any(indicator in font.fname.lower() for indicator in
                             ['chinese', 'cjk', 'noto', 'wqy']):
                        zh_fonts.append(font)

            if zh_fonts:
                # 使用找到的第一个中文字体
                font_prop = font_manager.FontProperties(
                    fname=zh_fonts[0].fname)
                font_found = True
            else:
                # 如果没找到特定的中文字体，尝试设置支持中文字体的通用属性
                plt.rcParams['font.sans-serif'] = [
                    'SimHei', 'DejaVu Sans', 'Liberation Sans',
                    'Noto Sans CJK SC'
                ]
                plt.rcParams[
                    'axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
                font_prop = font_manager.FontProperties()
    except:
        font_prop = font_manager.FontProperties()

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置标签
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # 使用字体属性设置标签
    ax.set_xticklabels(class_names,
                       rotation=45,
                       ha='right',
                       fontproperties=font_prop)
    ax.set_yticklabels(class_names, fontproperties=font_prop)

    # 在每个格子中显示数值
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j,
                    i,
                    format(cm_normalized[i, j], fmt),
                    ha="center",
                    va="center",
                    fontproperties=font_prop,
                    color="white" if cm_normalized[i, j] > thresh else "black")

    # 设置标题和轴标签（使用字体属性）
    ax.set_title("归一化混淆矩阵", fontproperties=font_prop)
    ax.set_xlabel("预测标签", fontproperties=font_prop)
    ax.set_ylabel("真实标签", fontproperties=font_prop)

    # 自动调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


from PIL import Image
def cacu_metrics(preds,labels,classes,save_dir):
    MIoU, F1, Kappa, Acc, cm = metrics_print_version(
            np.concatenate([p.ravel() for p in preds]),
            np.concatenate([p.ravel() for p in labels]).ravel(), classes)

    plot_confusion_matrix(cm, classes,
                          os.path.join(save_dir, "confusion_matrix.png"))

    # 构建详细指标字典
    detailed_metrics = {"MIoU": MIoU, "F1": F1, "Kappa": Kappa, "Acc": Acc}

    return detailed_metrics
def loaddata(folder_path, extension=".tif"):
    # 1. 将字符串路径转为 Path 对象
    folder = Path(folder_path)

    # 2. 获取该文件夹下所有指定后缀的影像路径 (返回一个列表)
    # 如果你的影像包含在子文件夹中，请把 .glob 换成 .rglob (递归搜索)
    image_paths = list(folder.glob(f"*{extension}"))

    if not image_paths:
        print(f"警告：在 {folder_path} 中没有找到任何 {extension} 文件！")
        return

    print(f"总共找到 {len(image_paths)} 张影像，准备处理...")

    # 3. 将路径列表放入 tqdm 中，自动生成进度条
    # desc: 进度条前面的描述文字
    # unit: 进度条后面的单位
    masks = []
    for img_path in tqdm(image_paths, desc="读取进度", unit="张"):

        # --- 在这里执行你的单张影像读取或处理逻辑 ---
        try:

            mask = Image.open(img_path)
            mask = np.array(mask)
            mask = torch.tensor(mask).unsqueeze(0)
            masks.append(mask)

        except Exception as e:
            # 使用 tqdm.write 代替 print，防止打印出的报错信息打乱进度条的排版
            tqdm.write(f"读取文件 {img_path.name} 时出错: {e}")

    return masks
def main(predfolder,labelfolder,numclass,savedir):
    preds = loaddata(predfolder)
    labels = loaddata(labelfolder)
    cacu_metrics(preds,labels,numclass,savedir)


if __name__ == "__main__":

    # predfolder = r'E:\segdino-main\segdino-main\runs\segdino_l_512_csgo2\test_vis'
    # labelfolder = r'E:\segdino-main\segdino-main\segdata\csgo2\test\mask'
    # numclass = ['其他', '森林', '草地', '农地']
    # savedir = r'E:\result_csgo'
    # main(predfolder,labelfolder,numclass,savedir)

    predfolder = r'E:\output\TALANDCOVER'
    labelfolder = r'E:\segdino-main\segdino-main\segdata\TALANDCOVER\test\labels'
    numclass = ["裸地退化", "草原", "异质农业区", "茂密森林", "水体", "建成区"]
    savedir = r'E:\output'
    main(predfolder, labelfolder, numclass, savedir)
