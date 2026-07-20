# MS_ROOT_DIR = r"E:\segdino-main\segdino-main\segdata\TALANDCOVER"

MS_ROOT_DIR = r"E:\segdino-main\segdino-main\segdata\csgo2"
def get_labels(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Please specify a dataset")

    elif dataset_name == "csgo1":
        labels = [
            "其他", "森林", "草地", "湿地"
        ]
    elif dataset_name == "csgo2":
        labels = [
            "其他", "森林", "草地", "农田"
        ]
    elif dataset_name == "TALANDCOVER":
        labels = [
            "裸地退化", "草原", "异质农业区", "茂密森林", "水体", "建成区"
        ]

    return labels
