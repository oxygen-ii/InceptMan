# InceptMan

InceptMan is a deep learning model designed for mandible reconstruction, as proposed in our paper, "[InceptMan: An InceptionNeXt-Based Architecture for End-to-End Mandible Reconstruction](https://ieeexplore.ieee.org/document/11048542)" The input is a binary volumetric incomplete mandible, and the output is a binary volumetric complete mandible.

![alt text](https://github.com/oxygen-ii/InceptMan/blob/main/image/model.png?raw=true)

# Requirements

Use `pip` to install the requirements as follows:
```
!pip install -r requirements.txt
```

# Train
We use batch size of 2 by default and we show how to train models with 2 GPUs.

```
python Train.py --dataset_path */dataset  --save_path */save_model
```

# Validation

# Mandible cutting algorithm

# Citation

Please refer to our full manuscript in IEEE Access. If you use the model, you can cite it with the following bibtex.

```
@ARTICLE{11048542,
  author={Kamboonsri, Nattapon and Tantisereepatana, Natdanai and Achakulvisut, Titipat and Vateekul, Peerapon},
  journal={IEEE Access}, 
  title={InceptMan: An InceptionNeXt-Based Architecture for End-to-End Mandible Reconstruction}, 
  year={2025},
  volume={13},
  number={},
  pages={108968-108983},
  keywords={Image reconstruction;Pipelines;Implants;Surgery;Computer architecture;Transformers;Computed tomography;Image segmentation;Deep learning;Three-dimensional displays;Deep learning;volumetric shape completion;mandible reconstruction;automated pipeline;virtual surgical planning},
  doi={10.1109/ACCESS.2025.3582504}
}
```
