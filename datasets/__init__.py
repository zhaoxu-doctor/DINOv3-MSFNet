import os

from configs.common_cfg import MS_ROOT_DIR
from .CSGO_dataset import csgo_Dataset, csgo_Dataset_test, TALANDCOVER_Dataset, TALANDCOVER_Dataset_test


def build_dataset(dataset_name, data_type="test", **kwargs):
    model_name = kwargs.get("model_name")
    backbone_type = kwargs.get("backbone_type")
    # if model_name == "DINOv3" and backbone_type in [
    #         "dinov3_vitl16", "dinov3_vit7b16"
    # ]:
    #     normalize_type = "geo"
    # else:
    #     normalize_type = "common"
    normalize_type = "geo"

    if dataset_name == "csgo1" or dataset_name == "csgo2":
        train_dir0 = f"{MS_ROOT_DIR}/train"
        val_dir0 = f"{MS_ROOT_DIR}/val"
        test_dir0 = f"{MS_ROOT_DIR}/test"
        train_dir = f"{MS_ROOT_DIR}/train/image"
        val_dir= f"{MS_ROOT_DIR}/val/image"
        test_dir = f"{MS_ROOT_DIR}/test/image"
        # 读取文件夹中所有tif文件名

        if data_type == "test":
            data_dir = test_dir0 + "/image/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(test_dir) if f.endswith(".tif")]
        elif data_type == "val":
            data_dir = val_dir0 + "/image/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(val_dir) if f.endswith(".tif")]
        else:
            data_dir = train_dir0 + "/image/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(train_dir) if f.endswith(".tif")]

        if data_type == "test":
            label_dir = test_dir0 + "/mask/{}.tif"
        elif data_type == "val":
            label_dir = val_dir0 + "/mask/{}.tif"
        else:
            label_dir = train_dir0 + "/mask/{}.tif"

        if data_type == "train":
            out = csgo_Dataset(ids=ids,
                         data_dir=data_dir,
                         label_dir=label_dir,
                         data_type=data_type,
                         window_size=kwargs.get("window_size", (512, 512)),
                         normalize_type=normalize_type)
        else:
            out = csgo_Dataset_test(ids=ids,
                            data_dir=data_dir,
                            label_dir=label_dir,
                            data_type=data_type,
                            window_size=kwargs.get("window_size", (512, 512)),
                            normalize_type=normalize_type)
        return out
    elif dataset_name == "TALANDCOVER":

        train_dir0 = f"{MS_ROOT_DIR}/train"
        val_dir0 = f"{MS_ROOT_DIR}/val"
        test_dir0 = f"{MS_ROOT_DIR}/test"
        train_dir = f"{MS_ROOT_DIR}/train/images"
        val_dir = f"{MS_ROOT_DIR}/val/images"
        test_dir = f"{MS_ROOT_DIR}/test/images"
        # 读取文件夹中所有tif文件名

        if data_type == "test":
            data_dir = test_dir0 + "/images/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(test_dir) if f.endswith(".tif")]
        elif data_type == "val":
            data_dir = val_dir0 + "/images/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(val_dir) if f.endswith(".tif")]
        else:
            data_dir = train_dir0 + "/images/{}.tif"
            ids = [f.split(".")[0] for f in os.listdir(train_dir) if f.endswith(".tif")]

        if data_type == "test":
            label_dir = test_dir0 + "/labels/{}.tif"
        elif data_type == "val":
            label_dir = val_dir0 + "/labels/{}.tif"
        else:
            label_dir = train_dir0 + "/labels/{}.tif"
        if data_type == "train":
            out = TALANDCOVER_Dataset(ids=ids,
                         data_dir=data_dir,
                         label_dir=label_dir,
                         data_type=data_type,
                         window_size=kwargs.get("window_size", (128, 128)),
                         normalize_type=normalize_type)
        else:
            out = TALANDCOVER_Dataset_test(ids=ids,
                            data_dir=data_dir,
                            label_dir=label_dir,
                            data_type=data_type,
                            window_size=kwargs.get("window_size", (128, 128)),
                            normalize_type=normalize_type)
        return out


