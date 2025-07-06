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
To evaluate our model, run:

```
python Inference.py --dataset_path */dataset  --save_path */save_model --output_path */output
```
# Mandible cutting algorithm

Our automatic pipeline consists of two main components:

Automated Cutting Pipeline – used to segment the mandible from the input volume using a custom Mandible Cutting Algorithm module.

Reconstruction Pipeline – utilizes a deep learning-based mandible reconstruction model to generate a complete mandible from the segmented input.

![alt text](https://github.com/oxygen-ii/InceptMan/blob/main/image/pipeline.png?raw=true)

💡 Example usage for the Mandible Cutting Algorithm module is provided in this.

```
python process.py --template_mandible_path */template_mandible.nii.gz  --binary_volumetric_mandibles_path */mandible.nii.gz,
                  --binary_volumetric_healthy_defective_mandible_path */healthy&defective mandible.nii.gz, --output_path *output.nii.gz
```

⚠️ Due to a Non-Disclosure Agreement (NDA), we are unable to share the pretrained weights of our segmentation and reconstruction models, as well as the cutting template used in this study.

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
