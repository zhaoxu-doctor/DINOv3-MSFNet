#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 label 的空间覆盖范围和类别像素划分遥感分割数据集。

依赖：pip install numpy rasterio pyproj scipy shapely tqdm
修改下方配置，然后运行：python spatial_split_from_labels.py
原始文件只读；RGB 的地理信息不参与任何判断。
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from tqdm import tqdm


# ============================ 请修改这里 ============================
IMAGE_DIR = Path(r"D:\csgo\csgo2\image")
LABEL_DIR = Path(r"D:\csgo\csgo2\mask")
OUTPUT_DIR = Path(r"D:\csgo\csgo2\new")

RATIOS = (0.8, 0.1, 0.1)       # train / val / test；是优化目标，非硬性精确比例
BUFFER_M = 500.0               # 不同集合 patch 的实际覆盖范围间距，米
BLOCK_SIZE_M = None            # 米；None 根据范围自动估算，推荐按结果调整
TARGET_BLOCKS = 40             # 仅用于自动估算 BLOCK_SIZE_M
BALANCE_CLASSES = None         # 如 [1, 2, 3]；None 为全部有效类别
IGNORE_VALUES = (255,)         # NoData 另外自动排除
SEED = 42
RESTARTS = 8                  # 多次优化空间块分配，选择目标函数较好的方案
COPY_FILES = True             # False 时仅输出 CSV 和统计，不复制影像
# ===================================================================

SPLITS = ("train", "val", "test")
EXTENSIONS = {".tif", ".tiff"}


def input_files(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    result = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            if p.name in result:
                raise ValueError(f"重复文件名: {p.name}")
            result[p.name] = p
    if not result:
        raise ValueError(f"文件夹里没有 tif/tiff: {folder}")
    return result


def verify_paths(images: dict, labels: dict) -> None:
    if set(images) != set(labels):
        a = sorted(set(labels) - set(images))[:8]
        b = sorted(set(images) - set(labels))[:8]
        raise ValueError(f"文件名必须完全一致；缺少 RGB: {a}；缺少 label: {b}")
    out = OUTPUT_DIR.resolve()
    if out == IMAGE_DIR.resolve() or out == LABEL_DIR.resolve():
        raise ValueError("输出目录不能是输入目录")
    if out.is_relative_to(IMAGE_DIR.resolve()) or out.is_relative_to(LABEL_DIR.resolve()):
        raise ValueError("输出目录不能位于输入目录之内")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免混入旧结果，停止: {out}")


def metric_crs_for_first_label(path: Path) -> CRS:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"label 无 CRS: {path}")
        x, y = src.transform * (src.width / 2, src.height / 2)
        lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)
    if not np.isfinite([lon, lat]).all() or not (-80 <= lat <= 84):
        raise ValueError("自动 UTM 仅支持纬度 -80° 至 84°，请检查 label CRS/坐标")
    zone = max(1, min(60, int(math.floor((lon + 180) / 6) + 1)))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def read_samples(images: dict, labels: dict, metric_crs: CRS):
    samples = []
    transformers = {}
    ignore = set(IGNORE_VALUES)
    for name in tqdm(sorted(labels), desc="读取 labels 坐标及类别"):
        with rasterio.open(labels[name]) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError(f"label 必须是带 CRS 的单波段 GeoTIFF: {name}")
            if src.transform.is_identity:
                raise ValueError(f"label 是默认 identity transform，无法计算真实空间位置: {name}")
            if not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                raise ValueError(f"类别标签必须是整型: {name}, dtype={src.dtypes[0]}")
            arr = src.read(1, masked=True)
            valid = ~np.ma.getmaskarray(arr)
            vals = np.asarray(arr.data)[valid]
            if ignore:
                vals = vals[~np.isin(vals, list(ignore))]
            values, counts = np.unique(vals, return_counts=True)
            hist = {int(k): int(v) for k, v in zip(values, counts)}

            key = src.crs.to_wkt()
            if key not in transformers:
                transformers[key] = Transformer.from_crs(src.crs, metric_crs, always_xy=True)
            transformer = transformers[key]
            # 细分四条边后投影，避免只投影四角造成明显的边界误差。
            n = 8
            border = ([(t * src.width / n, 0) for t in range(n)]
                      + [(src.width, t * src.height / n) for t in range(n)]
                      + [(src.width - t * src.width / n, src.height) for t in range(n)]
                      + [(0, src.height - t * src.height / n) for t in range(n)])
            coords = [transformer.transform(*(src.transform * (c, r))) for c, r in border]
            if not np.isfinite(coords).all():
                raise ValueError(f"label 坐标转换失败: {name}")
            polygon = Polygon(coords)
            if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                raise ValueError(f"label 覆盖范围无效: {name}")
            # 平均边界点作为外接圆中心；半径涵盖所有边界点。
            xy = np.mean(coords, axis=0)
            radius = float(np.max(np.linalg.norm(np.asarray(coords) - xy, axis=1)))
            samples.append({"name": name, "image": images[name], "label": labels[name],
                            "xy": xy, "radius": radius, "polygon": polygon, "hist": hist})
    return samples


def make_blocks(samples):
    xy = np.array([s["xy"] for s in samples])
    extent = np.ptp(xy, axis=0)
    patch_diameter = float(np.median([2 * s["radius"] for s in samples]))
    if BLOCK_SIZE_M is None:
        # 对狭长研究区也保持足够多的块；用户可在配置中固定此参数。
        block_size = max(patch_diameter * 1.5, math.sqrt(max(float(np.prod(extent)), 1.0) / TARGET_BLOCKS))
    else:
        block_size = float(BLOCK_SIZE_M)
    if block_size <= 0:
        raise ValueError("BLOCK_SIZE_M 必须大于 0")
    cells = np.floor((xy - xy.min(axis=0)) / block_size).astype(np.int64)
    keys = sorted({tuple(p) for p in cells})
    key_ids = {key: i for i, key in enumerate(keys)}
    for s, cell in zip(samples, cells):
        s["block"] = key_ids[tuple(cell)]
        s["cell"] = tuple(int(x) for x in cell)
    if len(keys) < 3:
        raise ValueError(f"只有 {len(keys)} 个空间块；请减小 BLOCK_SIZE_M ({block_size:.1f} m)")
    return block_size, keys


def near_pairs(samples):
    """找出实际覆盖范围之间距离小于缓冲距离的 patch 对。"""
    coords = np.array([s["xy"] for s in samples])
    radii = np.array([s["radius"] for s in samples])
    tree = cKDTree(coords)
    max_r = float(radii.max())
    pairs = []
    eps = 1e-6
    for i, s in enumerate(tqdm(samples, desc="建立空间缓冲邻接")):
        near = tree.query_ball_point(coords[i], radii[i] + max_r + BUFFER_M + eps)
        for j in near:
            if j <= i or samples[j]["block"] == s["block"]:
                continue
            if np.linalg.norm(coords[i] - coords[j]) > radii[i] + radii[j] + BUFFER_M + eps:
                continue
            distance = s["polygon"].distance(samples[j]["polygon"])
            # BUFFER_M=0 时仍须排除不同集合的重叠 patch。
            if distance < BUFFER_M + eps or (BUFFER_M == 0 and distance <= eps):
                pairs.append((i, j))
    return pairs


def aggregate(samples, classes, n_blocks):
    counts = np.zeros(n_blocks, dtype=np.int64)
    pixels = np.zeros((n_blocks, len(classes)), dtype=np.float64)
    for s in samples:
        b = s["block"]
        counts[b] += 1
        pixels[b] += [s["hist"].get(c, 0) for c in classes]
    return counts, pixels


def score(assignment, counts, pixels, edge_a, edge_b, edge_w, targets):
    set_n = np.bincount(assignment, weights=counts, minlength=3)
    set_p = np.array([pixels[assignment == i].sum(axis=0) for i in range(3)])
    frac_n = set_n / max(float(set_n.sum()), 1.0)
    count_loss = np.mean((frac_n - targets) ** 2 / (targets + 0.05))
    total_p = set_p.sum(axis=0)
    valid = total_p > 0
    class_loss = 0.0
    if valid.any():
        frac_p = set_p[:, valid] / total_p[valid]
        class_loss = float(np.mean((frac_p - targets[:, None]) ** 2 / (targets[:, None] + 0.05)))
    # 额外鼓励同一集合的空间块相邻，降低边界剔除量。
    cut = float(edge_w[assignment[edge_a] != assignment[edge_b]].sum())
    cut_loss = cut / max(float(edge_w.sum()), 1.0)
    return float(count_loss + class_loss + 0.08 * cut_loss)


def optimize_blocks(counts, pixels, pairs, samples, seed):
    n = len(counts)
    targets = np.asarray(RATIOS, dtype=float)
    edges = Counter(tuple(sorted((samples[i]["block"], samples[j]["block"]))) for i, j in pairs)
    edge_a = np.fromiter((a for a, b in edges), dtype=np.int64)
    edge_b = np.fromiter((b for a, b in edges), dtype=np.int64)
    edge_w = np.fromiter(edges.values(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_a, best_loss = None, float("inf")
    for run in range(RESTARTS):
        a = np.zeros(n, dtype=np.int8)
        order = rng.permutation(n)
        # 初始为 train，两个不同空间块分别设为 val/test。
        a[order[0]] = 1
        a[order[1]] = 2
        for _ in range(12):
            changed = False
            for b in rng.permutation(n):
                before = int(a[b])
                if np.count_nonzero(a == before) == 1:
                    continue
                losses = []
                for choice in range(3):
                    a[b] = choice
                    losses.append(score(a, counts, pixels, edge_a, edge_b, edge_w, targets))
                choice = int(np.argmin(losses))
                a[b] = choice
                changed |= choice != before
            if not changed:
                break
        loss = score(a, counts, pixels, edge_a, edge_b, edge_w, targets)
        if loss < best_loss:
            best_a, best_loss = a.copy(), loss
    if best_a is None or len(set(best_a)) < 3:
        raise RuntimeError("无法形成三个非空集合，请减小 BLOCK_SIZE_M 或 BUFFER_M")
    return best_a, edges


def apply_buffer(samples, pairs, assignment):
    """冲突图的确定性贪心顶点覆盖：优先舍弃冲突较多的边界 patch。"""
    groups = np.array([int(assignment[s["block"]]) for s in samples], dtype=np.int8)
    adj = [set() for _ in samples]
    for i, j in pairs:
        if groups[i] != groups[j]:
            adj[i].add(j)
            adj[j].add(i)
    removed = np.zeros(len(samples), dtype=bool)
    # 小集合样本更昂贵，尽可能保留 val/test；每次重新计算未解决的冲突度。
    def entry(i):
        degree = len(adj[i])
        cost = 1 if groups[i] == 0 else 3
        return (-degree / cost, -degree, i, degree)

    queue = [entry(i) for i in range(len(samples)) if adj[i]]
    heapq.heapify(queue)
    while queue:
        _, _, i, expected = heapq.heappop(queue)
        if removed[i] or len(adj[i]) != expected or expected == 0:
            continue
        removed[i] = True
        for j in adj[i]:
            adj[j].discard(i)
            if adj[j] and not removed[j]:
                heapq.heappush(queue, entry(j))
        adj[i].clear()
    kept = [int(np.sum((groups == k) & ~removed)) for k in range(3)]
    if min(kept) == 0:
        raise RuntimeError(
            f"缓冲后集合为空，保留数量={kept}；尝试减小 BUFFER_M/增大研究区或调整 BLOCK_SIZE_M"
        )
    # 重新扫描全部空间冲突；若未解决，绝不输出貌似独立的划分。
    if any(not removed[i] and not removed[j] and groups[i] != groups[j] for i, j in pairs):
        raise AssertionError("缓冲区校验失败，存在不同集合的近邻 patch")
    return groups, removed


def save_results(samples, keys, classes, block_size, metric_crs, groups, removed):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, hist = [], {k: defaultdict(int) for k in (*SPLITS, "buffer")}
    counts = Counter()
    for i, s in enumerate(tqdm(samples, desc="输出清单及数据")):
        split = "buffer" if removed[i] else SPLITS[int(groups[i])]
        counts[split] += 1
        for cls, count in s["hist"].items():
            hist[split][cls] += count
        if COPY_FILES and split != "buffer":
            for kind, source in (("images", s["image"]), ("labels", s["label"])):
                target = OUTPUT_DIR / split / kind / s["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        rows.append({"filename": s["name"], "split": split, "original_split": SPLITS[int(groups[i])],
                     "block_id": s["block"], "block_x": s["cell"][0], "block_y": s["cell"][1],
                     "label_center_x_m": f"{s['xy'][0]:.3f}", "label_center_y_m": f"{s['xy'][1]:.3f}"})
    with (OUTPUT_DIR / "split_files.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "block_assignment.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["block_id", "block_x", "block_y", "original_split"])
        for b, (x, y) in enumerate(keys):
            writer.writerow([b, x, y, SPLITS[int(groups[next(i for i, s in enumerate(samples) if s['block'] == b)])]])
    kept = sum(counts[k] for k in SPLITS)
    summary = {"total": len(samples), "metric_crs": metric_crs.to_string(),
               "buffer_m": BUFFER_M, "block_size_m": block_size,
               "target_ratios": dict(zip(SPLITS, RATIOS)), "balance_classes": classes,
               "counts": {k: counts[k] for k in (*SPLITS, "buffer")},
               "retained_ratios": {k: counts[k] / kept for k in SPLITS},
               "class_pixels": {k: {str(c): hist[k][c] for c in classes} for k in (*SPLITS, "buffer")},
               "class_retained_ratios": {str(c): {
                   k: hist[k][c] / max(sum(hist[x][c] for x in SPLITS), 1) for k in SPLITS
               } for c in classes}}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n最终数量:", summary["counts"])
    print("保留样本比例:", summary["retained_ratios"])
    print("按类别统计（保留样本中的像素比例）:", summary["class_retained_ratios"])
    print("输出目录:", OUTPUT_DIR)


def main():
    if len(RATIOS) != 3 or any(r <= 0 for r in RATIOS) or not np.isclose(sum(RATIOS), 1):
        raise ValueError("RATIOS 必须为三个正数，且总和为 1")
    if BUFFER_M < 0 or RESTARTS < 1 or TARGET_BLOCKS < 3:
        raise ValueError("BUFFER_M、RESTARTS 或 TARGET_BLOCKS 配置无效")
    images, labels = input_files(IMAGE_DIR), input_files(LABEL_DIR)
    verify_paths(images, labels)
    metric_crs = metric_crs_for_first_label(labels[sorted(labels)[0]])
    print(f"匹配 {len(labels)} 对文件；全部空间坐标仅取自 labels；距离坐标系: {metric_crs}")
    samples = read_samples(images, labels, metric_crs)
    discovered = sorted({c for s in samples for c in s["hist"]})
    classes = discovered if BALANCE_CLASSES is None else list(dict.fromkeys(BALANCE_CLASSES))
    if not classes or any(c not in discovered for c in classes):
        raise ValueError(f"均衡类别 {classes} 无效；实际有效类别: {discovered}")
    block_size, keys = make_blocks(samples)
    print(f"空间块 {len(keys)} 个，块边长约 {block_size:.1f} m；类别: {classes}")
    counts, pixels = aggregate(samples, classes, len(keys))
    pairs = near_pairs(samples)
    print(f"不同块之间的潜在缓冲冲突 {len(pairs)} 对")
    assignment, _ = optimize_blocks(counts, pixels, pairs, samples, SEED)
    groups, removed = apply_buffer(samples, pairs, assignment)
    save_results(samples, keys, classes, block_size, metric_crs, groups, removed)


if __name__ == "__main__":
    main()
