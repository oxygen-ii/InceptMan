import os
from model import InceptMan_m

import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset, download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, TverskyLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, AttentionUnet, BasicUNetPlusPlus, SwinUNETR, UNETR, ViT
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    Flipd,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SaveImage,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
)
from monai.transforms import SaveImage
from monai.utils import first, set_determinism
import numpy as np
import onnxruntime
from tqdm import tqdm
import glob
import shutil
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="define dataset folder")
    parser.add_argument("--save_path", type=str, help="define a saved folder (model)")
    parser.add_argument("--output_path", type=str, help="define a saved folder (output)")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Load Data
    save_dir = args.save_path                                                                                               
    data_folder_dir = args.dataset_path                                                                                   
    data_dir = [os.path.join(data_folder_dir, name) for name in sorted(os.listdir(data_folder_dir))]
    test_images = [sorted(glob.glob(os.path.join(image))) for image in data_dir]
    test_files = [{"image": image} for image in test_images]
    
    roi_size = (160, 128, 128) 
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = InceptMan_m(square_kernel_size=3, band_kernel_size=11).to(device)                                             
    
    class ConvertToCuttingChannel(MapTransform):
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = (d[key] == 1).float()
            return d

    class ConvertToMandibleChannel(MapTransform):
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = torch.logical_or(d[key] == 1, d[key] == 2).float()
            return d
    
    test_org_transforms = Compose(
        [
            LoadImaged(keys=["image"],ensure_channel_first=True,image_only=True,dtype=torch.float),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image", margin=10),
            Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=("nearest")),
            SpatialPadd(keys=["image"], spatial_size=roi_size),
            ConvertToCuttingChannel(keys="image"),
        ]
    )

    test_org_ds = Dataset(data=test_files, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        ]
    )
    
    state_dict = torch.load(os.path.join(save_dir, "InceptMan_best_metric_model.pth"), map_location='cuda:0')
    # Remove 'module.' prefix if exists
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.'
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)

    model.eval()
    throughput_infer = []
    with torch.no_grad():
        for test_data in test_org_loader:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            test_inputs = test_data["image"].to(device) 
            sw_batch_size = 1
            start.record()
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            throughput = 1 / latency
            throughput_infer.append(throughput)
            print(f"throughput(infer): {throughput}")
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            # Create the output directory if it doesn't exist
            output_dir = args.output_path           
            os.makedirs(output_dir, exist_ok=True)

            # Create a 3D tensor (e.g., a simple example tensor)
            tensor = torch.argmax(test_data[0]["pred"], dim=0).detach().cpu()
            #tensor = torch.sigmoid(test_data[0]["pred"], dim=0).detach().cpu()
                    
            # Metadata for saving the NIfTI file
            metadata = {
                        "filename_or_obj": os.path.join(output_dir, "example.nii.gz")
                    }
                    
                    # Initialize the SaveImage transform
            saver = SaveImage(output_dir=output_dir, output_postfix="InceptMan", output_ext=".nii.gz", resample=False)        
                
                    # Save the tensor as a NIfTI file
            saver(tensor, meta_data=metadata)
                    
            print(f"NIfTI file saved in {output_dir}")

if __name__ == "__main__":
    main()
