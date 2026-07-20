from .DINOv3_MSFNet import get_cfg as get_DINOv3_cfg


def get_cfg(model_name=None, dataset_name=None, **kwargs):
    if model_name is None:
        raise ValueError("Model name must be specified")
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")
    if 'DINOv3_Baseline' == model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)
    if 'DINOv3_ONLYFEM' == model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)
    if 'DINOv3_ONLYCSFM' == model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)
    if 'DINOv3_FEM_CSFM' == model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)

    return cfg
