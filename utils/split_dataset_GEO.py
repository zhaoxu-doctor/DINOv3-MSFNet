# -*- coding: utf-8 -*-
"""
基于空间聚类 + 类别均衡的数据集划分脚本

输入:
    mask_dir  : 包含0,1,2,3标签tif的mask文件夹
    image_dir : 与mask一一对应的image文件夹

输出:
    output_dir/train/{mask,image}
    output_dir/val/{mask,image}
    output_dir/test/{mask,image}
    output_dir/split_files.csv

依赖:
    pip install rasterio numpy scikit-learn tqdm
"""

import os
import random
import shutil
from collections import defaultdict

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


def collect_mask_info(mask_dir):
    """读取mask文件夹中所有tif，获取坐标、尺寸、类别像素计数"""
    mask_files = [f for f in os.listdir(mask_dir)
                  if f.lower().endswith(('.tif', '.tiff'))]

    infos = []
    for fname in tqdm(mask_files, desc="Reading masks"):
        mask_path = os.path.join(mask_dir, fname)
        try:
            with rasterio.open(mask_path) as src:
                band = src.read(1)
                height, width = band.shape
                transform = src.transform

                # 影像中心点地理坐标
                col_center = (width - 1) / 2.0
                row_center = (height - 1) / 2.0
                x, y = transform * (col_center, row_center)

                # 统计0,1,2,3各类别像素数
                flat = band.ravel()
                counts = np.bincount(flat, minlength=4)[:4].astype(np.int64)

                infos.append({
                    "name": fname,
                    "mask_path": mask_path,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "counts": counts,
                })
        except Exception as e:
            print(f"警告：无法读取 {fname}: {e}")

    if not infos:
        raise RuntimeError(f"在 {mask_dir} 中没有找到可用的tif文件")

    return infos


def assign_spatial_clusters(infos, n_clusters=None, random_state=42):
    """基于中心坐标进行KMeans空间聚类"""
    coords = np.array([[p["x"], p["y"]] for p in infos])

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    n = len(coords_scaled)
    if n_clusters is None:
        # 自动选择簇数量，尽量保证足够多的空间块用于比例分配
        n_clusters = max(10, min(30, int(np.sqrt(n * 2))))
    if n_clusters > n:
        n_clusters = n

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords_scaled)

    for info, label in zip(infos, labels):
        info["cluster"] = int(label)

    return infos


def find_image_path(image_dir, mask_name):
    """在image目录中找到与mask同名的文件；允许扩展名不同但stem相同"""
    exact = os.path.join(image_dir, mask_name)
    if os.path.isfile(exact):
        return exact

    stem, _ = os.path.splitext(mask_name)
    for f in os.listdir(image_dir):
        if os.path.splitext(f)[0] == stem:
            return os.path.join(image_dir, f)
    return None


def aggregate_clusters(infos, image_dir):
    """将每个簇的影像路径、类别计数聚合起来"""
    cluster_buckets = defaultdict(lambda: {
        "names": [],
        "mask_paths": [],
        "image_paths": [],
        "counts": None,
        "n_images": 0,
    })

    for p in infos:
        cid = p["cluster"]
        img_path = find_image_path(image_dir, p["name"])
        if img_path is None:
            raise FileNotFoundError(
                f"image目录中找不到 {p['name']} 对应的影像，请检查数据"
            )

        b = cluster_buckets[cid]
        b["names"].append(p["name"])
        b["mask_paths"].append(p["mask_path"])
        b["image_paths"].append(img_path)
        b["n_images"] += 1

        if b["counts"] is None:
            b["counts"] = p["counts"].copy()
        else:
            b["counts"] += p["counts"]

    cluster_data = []
    for cid in sorted(cluster_buckets.keys()):
        b = cluster_buckets[cid]
        cluster_data.append({
            "id": cid,
            "counts": b["counts"],
            "n_images": b["n_images"],
            "names": b["names"],
            "mask_paths": b["mask_paths"],
            "image_paths": b["image_paths"],
        })

    return cluster_data


def split_loss(assignment, cluster_data, targets):
    """损失函数：同时衡量影像数量比例和各类别像素比例"""
    n_sets = len(targets)
    K = len(cluster_data)

    total_counts = np.sum([d["counts"] for d in cluster_data], axis=0)
    total_images = sum(d["n_images"] for d in cluster_data)

    set_counts = np.zeros((n_sets, 4), dtype=np.float64)
    set_images = np.zeros(n_sets, dtype=float)

    for cid, d in enumerate(cluster_data):
        s = assignment[cid]
        set_counts[s] += d["counts"]
        set_images[s] += d["n_images"]

    loss = 0.0
    for s in range(n_sets):
        # 影像数量比例差异
        image_frac = set_images[s] / max(total_images, 1)
        loss += (image_frac - targets[s]) ** 2

        # 各类别像素数量与理想分配数量的差异（归一化）
        ideal_counts = total_counts * targets[s]
        diff = (set_counts[s] - ideal_counts) / np.maximum(total_counts, 1)
        loss += np.sum(diff ** 2)

    return loss


def initial_greedy_assignment(rng, cluster_data, targets):
    """贪心初始化：随机顺序下将每个簇放到当前损失最小的集合中"""
    n_sets = len(targets)
    n_clusters = len(cluster_data)
    assignment = [-1] * n_clusters

    total_counts = np.sum([d["counts"] for d in cluster_data], axis=0)
    total_images = sum(d["n_images"] for d in cluster_data)

    set_counts = np.zeros((n_sets, 4), dtype=np.float64)
    set_images = np.zeros(n_sets, dtype=float)

    order = list(range(n_clusters))
    rng.shuffle(order)

    for cid in order:
        d = cluster_data[cid]
        best_loss = np.inf
        best_set = 0

        for s in range(n_sets):
            temp_counts = set_counts.copy()
            temp_images = set_images.copy()
            temp_counts[s] += d["counts"]
            temp_images[s] += d["n_images"]

            loss = 0.0
            for ss in range(n_sets):
                image_frac = temp_images[ss] / max(total_images, 1)
                loss += (image_frac - targets[ss]) ** 2

                ideal_counts = total_counts * targets[ss]
                diff = (temp_counts[ss] - ideal_counts) / np.maximum(total_counts, 1)
                loss += np.sum(diff ** 2)

            if loss < best_loss:
                best_loss = loss
                best_set = s

        assignment[cid] = best_set
        set_counts[best_set] += d["counts"]
        set_images[best_set] += d["n_images"]

    return assignment


def optimize_assignment(cluster_data,
                        targets=(0.8, 0.1, 0.1),
                        n_restarts=50,
                        n_iter=1500,
                        random_state=42):
    """使用模拟退火优化簇到train/val/test的分配"""
    n_sets = len(targets)
    n_clusters = len(cluster_data)
    rng = random.Random(random_state)

    best_assignment = None
    best_loss = np.inf

    for restart in range(n_restarts):
        # 贪心初始化 + 随机重启
        assignment = initial_greedy_assignment(rng, cluster_data, targets)
        current_loss = split_loss(assignment, cluster_data, targets)

        # 模拟退火局部搜索
        temperature = 1.0
        for _ in range(n_iter):
            cid = rng.randrange(n_clusters)
            old_set = assignment[cid]
            new_set = rng.randrange(n_sets)

            if new_set == old_set:
                continue

            assignment[cid] = new_set
            new_loss = split_loss(assignment, cluster_data, targets)
            delta = new_loss - current_loss

            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current_loss = new_loss
            else:
                assignment[cid] = old_set

            temperature *= 0.999

        if current_loss < best_loss:
            best_loss = current_loss
            best_assignment = assignment.copy()

    return best_assignment, best_loss


def split_dataset(mask_dir,
                  image_dir,
                  output_dir,
                  n_clusters=None,
                  ratios=(0.8, 0.1, 0.1),
                  copy_files=True,
                  n_restarts=50):
    """主流程"""
    print("Step 1: 读取mask信息...")
    infos = collect_mask_info(mask_dir)
    print(f"共找到 {len(infos)} 个mask文件")

    print("Step 2: 空间聚类...")
    infos = assign_spatial_clusters(infos, n_clusters=n_clusters)

    print("Step 3: 聚合空间簇...")
    cluster_data = aggregate_clusters(infos, image_dir)
    print(f"空间簇数量: {len(cluster_data)}")

    print("Step 4: 优化train/val/test分配...")
    assignment, loss = optimize_assignment(
        cluster_data,
        targets=list(ratios),
        n_restarts=n_restarts
    )
    print(f"最优损失: {loss:.6f}")

    set_names = ["train", "val", "test"]
    set_map = {i: set_names[i] for i in range(len(ratios))}

    # 创建输出目录
    for sname in set_names:
        os.makedirs(os.path.join(output_dir, sname, "mask"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, sname, "image"), exist_ok=True)

    # 复制文件并统计
    stats = {
        sname: {"n_images": 0, "counts": np.zeros(4, dtype=np.int64)}
        for sname in set_names
    }
    file_records = []

    for cid, d in enumerate(cluster_data):
        s = assignment[cid]
        sname = set_map[s]

        stats[sname]["n_images"] += d["n_images"]
        stats[sname]["counts"] += d["counts"]

        for mask_path, img_path in zip(d["mask_paths"], d["image_paths"]):
            mask_name = os.path.basename(mask_path)
            img_name = os.path.basename(img_path)

            dst_mask = os.path.join(output_dir, sname, "mask", mask_name)
            dst_img = os.path.join(output_dir, sname, "image", img_name)

            if copy_files:
                shutil.copy2(mask_path, dst_mask)
                shutil.copy2(img_path, dst_img)

            file_records.append((mask_name, sname, cid))

    # 打印划分统计
    total_images = sum(stats[s]["n_images"] for s in set_names)
    total_counts = np.sum([stats[s]["counts"] for s in set_names], axis=0)

    print("\n========== 划分结果统计 ==========")
    for sname in set_names:
        n = stats[sname]["n_images"]
        cnt = stats[sname]["counts"]
        cnt_total = cnt.sum()
        ratio_str = ", ".join(
            [f"类{c}: {cnt[c] / max(cnt_total, 1) * 100:.2f}%" for c in range(4)]
        )
        print(f"{sname:5s} | 影像数: {n:5d} ({n / max(total_images, 1) * 100:5.1f}%) | "
              f"像素数: {cnt_total:10d} | {ratio_str}")

    print("\n全局类别比例:")
    global_ratio = ", ".join(
        [f"类{c}: {total_counts[c] / max(total_counts.sum(), 1) * 100:.2f}%" for c in range(4)]
    )
    print(global_ratio)

    # 保存划分列表
    list_path = os.path.join(output_dir, "split_files.csv")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("filename,set,cluster_id\n")
        for mask_name, sname, cid in file_records:
            f.write(f"{mask_name},{sname},{cid}\n")
    print(f"\n文件清单已保存到: {list_path}")

    return stats


def main():
    mask_dir = r"D:\csgo\TALANDCOVER\labels"
    image_dir = r"D:\csgo\TALANDCOVER\images"
    output_dir = r"D:\csgo\TALANDCOVER\split"

    split_dataset(
        mask_dir=mask_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        n_clusters=None,            # 自动选择空间簇数量；也可手动指定如15
        ratios=(0.8, 0.1, 0.1),     # train:val:test
        copy_files=True,            # True复制文件，False只生成清单
        n_restarts=50               # 优化次数，越大效果越好但耗时增加
    )


if __name__ == "__main__":
    main()