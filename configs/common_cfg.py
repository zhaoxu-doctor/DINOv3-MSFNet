# MS_ROOT_DIR = r"E:\segdino-main\segdino-main\segdata\TALANDCOVER"

MS_ROOT_DIR = r"E:\segdino-main\segdino-main\segdata\csgo2_discussion"
def get_labels(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Please specify a dataset")

    elif dataset_name == "csgo1":
        labels = [
            "BACKGROUND", "FOREST", "GRASSLAND", "WETLAND"
        ]
    elif dataset_name == "csgo2":
        labels = [
            "BACKGROUND", "FOREST", "GRASSLAND", "CROPLAND"
        ]
    elif dataset_name == "TALANDCOVER":
        labels = [
            "Bare-degraded lands", "Grasslands", "Heterogeneous agricultural areas", "Dense forest", "Water Bodies", "Built-up areas"
        ]

    return labels
