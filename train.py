import os
from model import InceptMan_m

import itk
import json
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset, download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss, TverskyLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, AttentionUnet, UNETR, SwinUNETR
from monai.networks.layers import Norm
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CopyItemsd,
    CropForegroundd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandAffined,
    Resized,
    SaveImage,
    SaveImaged,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
    ToDeviced,
)
from monai.transforms import SaveImage
from monai.utils import first, set_determinism
import numpy as np
import onnxruntime
from tqdm import tqdm
import glob
import shutil
import torch
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="define dataset folder")
    parser.add_argument("--save_path", type=str, help="define a saved folder")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Load Data
    data_folder_dir = args.dataset_path                                                                             
    data_dir = [os.path.join(data_folder_dir, name) for name in sorted(os.listdir(data_folder_dir))]
    save_dir = args.save_path                                                                                             
    train_images = [sorted(glob.glob(os.path.join(image))) for image in data_dir]
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_images)]
    train_size = int(len(data_dicts) * 0.8)
    train_files, val_files = data_dicts[:train_size] , data_dicts[train_size:]                                                             
    
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
        
    # Transforms
    roi_size = (160, 128, 128)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],ensure_channel_first=True,image_only=True,dtype=torch.float),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="label", margin=10),
            ConvertToCuttingChannel(keys="image"),
            ConvertToMandibleChannel(keys="label"),
            Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("nearest", "nearest")),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"],ensure_channel_first=True,image_only=True,dtype=torch.float),
            Orientationd(keys=["image", "label"], axcodes="RAS"),    
            CropForegroundd(keys=["image", "label"], source_key="label", margin=10),
            ConvertToCuttingChannel(keys="image"),
            ConvertToMandibleChannel(keys="label"),
            Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("nearest", "nearest")),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        ]
    )

    # Dataset
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=20)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=20)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=20)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=20)

    # Model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")             
    model = InceptMan_m(square_kernel_size=3, band_kernel_size=11)                                                 #####
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training
    max_epochs = 100                                                                                                                
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    throughput_train = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        epoch_loss = 0
        step = 0
        start.record()
        torch.cuda.reset_peak_memory_stats()

        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 20 == 0:                                                                                                      
              print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        throughput = train_size / latency
        epoch_loss_values.append(epoch_loss)
        throughput_train.append(throughput)
        print(f"throughput(train): {throughput}")
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
    
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
    
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(save_dir, "InceptMan_best_metric_model.pth"), _use_new_zipfile_serialization=False)  
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        
            # Saving
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"InceptMan_{epoch + 1}_metric_model.pth"), _use_new_zipfile_serialization=False)
            np.save(os.path.join(save_dir, "InceptMan_epoch_loss"), np.array(epoch_loss_values))                                                    
            np.save(os.path.join(save_dir, "InceptMan_metric_values"), np.array(metric_values))
            np.save(os.path.join(save_dir, "InceptMan_throughput_train"), np.array(throughput_train))
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.2f} s")
