# DINOv3-MSFNet
DINOv3-MSFNet: A Multi-Scale Feature Fusion Network for Land Cover Classification in Mountainous Forest Ecosystems

For reproducibility and further research, we provide the datasets and trained model weights used in our experiments.

## Datasets and Pre-trained Weights

The experimental datasets and model weights used in this work can be downloaded from Baidu Netdisk:

**Baidu Netdisk:**
 https://pan.baidu.com/s/16BAltaeQll8zJ28FHl-mLg?pwd=MSFN 提取码: MSFN 

The shared folder contains the following resources:

| Resource              | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| Dataset #1            | DXAL dataset used for comparative experiments and ablation studies              |
| Dataset #2            | XXAL dataset used for comparative experiments                                   |
| Dataset #3            | TALANDCOVER dataset used for comparative experiments                            |
| DINOV3-MSFNet weights | Trained weights of the proposed DINOV3-MSFNet on different datasets             |
| Ablation weights      | Model weights corresponding to the ablation experiments conducted on Dataset #1 |

In total, the shared resources include:

* Three datasets used in the comparative experiments.
* Trained weights of **DINOV3-MSFNet** on different datasets.
* Model weights for the ablation experiments conducted on **Dataset #1**.

These files are provided to facilitate reproducibility and fair comparison with our reported experimental results.

## Recommended Directory Structure

After downloading the datasets and model weights, the files can be organized as follows:

```text
DINOV3-MSFNet/
├── datasets/
│ ├── Dataset1/
│ ├── Dataset2/
│ └── Dataset3/
├── models/
├── utils/
├── train.py
├── test.py
└── README.md
```

The actual directory names can be adjusted according to the configuration files used in the project.

## Model Weights

The trained weights provided in the Baidu Netdisk folder correspond to the experimental results reported in our paper.

For evaluation, please select the checkpoint corresponding to the target dataset and modify the checkpoint path in the testing configuration accordingly.

The **Dataset #1** folder additionally contains checkpoints from the ablation experiments, which can be used to reproduce and verify the contribution of individual components of DINOV3-MSFNet.

## Reproducibility

To reproduce the experimental results reported in the paper:

1. Download the datasets and model weights from the link above.
2. Place the datasets in the corresponding dataset directories.
3. Select the appropriate DINOV3-MSFNet checkpoint for the target dataset.
4. Configure the dataset path and checkpoint path.
5. Run the evaluation script to obtain the corresponding experimental results.

For the ablation study, use **Dataset #1** together with the corresponding ablation checkpoints.

The complete citation information will be updated after publication.

## Acknowledgements

We sincerely thank the authors and contributors of the publicly available datasets and open-source projects used in this work.

## Contact

If you have any questions regarding the code, datasets, model weights, or experimental reproduction, please feel free to open an issue in this repository.

# Train this model using your custom dataset

```
python train.py 
```


# Test this model using your custom dataset
```
python test.py 
```


```
@article{Zhao2026DINOv3-MSFNet,
  title={DINOv3-MSFNet: A Multi-Scale Feature Fusion Network for Land Cover Classification in Mountainous Forest Ecosystems},
  author={Xu Zhao, Lizhi Liu and Qiuliang Zhang },
  journal={ISPRS International Journal of Geo-Information},
  year={2026}
}.
```
