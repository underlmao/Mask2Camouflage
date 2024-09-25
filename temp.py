

import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data.detection_utils import read_image
import cv2
from mask2camouflage import add_dcnet_config
# Set up logger
from detectron2.projects.deeplab import add_deeplab_config
import numpy as np
import time

setup_logger()

# Function to set up Mask2Former config
def setup_cfg(config_file, weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_dcnet_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.set_new_allowed(True)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda:0"  # Use GPU
    cfg.freeze()
    return cfg

# Function to generate heatmap
def save_heatmap(mask, output_path):
    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, heatmap)
    
# Function to preprocess and predict instance segmentation
def predict_instance_segmentation(input_folder, output_folder, cfg):
    os.makedirs(output_folder, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    
    
    total_time = 0
    num_images = 0
    
    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            img = read_image(img_path, format="BGR")
            
            start_time = time.time()
            outputs = predictor(img)
            total_time += time.time() - start_time
            num_images += 1
            
            
            
            # Visualize the predictions
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
            # output_img_name = f"{os.path.basename(input_folder)}_{img_name}_segmented.png"
            # output_path = os.path.join(output_folder, output_img_name)
            # cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
            # print(f"Saved {output_path}")
            output_path = os.path.join(output_folder, f"{img_name}_segmented.png")
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
            print(f"Saved {output_path}")
    avg_fps = num_images / total_time if total_time > 0 else 0
    print(f"Processed {num_images} images in {total_time:.2f} seconds. Average FPS: {avg_fps:.2f}")
 
def predict_heat_map(input_folder, output_folder, cfg):
    os.makedirs(output_folder, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    
    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            img = read_image(img_path, format="BGR")
            outputs = predictor(img)
            
            # Assuming output['instances'].pred_masks contains the masks
            masks = outputs['instances'].pred_masks.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_path = os.path.join(output_folder, f"{img_name.split('.')[0]}.png")
                save_heatmap(mask, mask_path)
                print(f"Saved {mask_path}")


# Paths
input_folder = '/nfs/home/haiphung106/HAINet/pool/COD10K/Test_Image_CAM/'
# input_folder = '/nfs/home/haiphung106/HAINet/pool/NC4K/test/image'

folder = "hainet-0701-DCNetGlob-MS-Zoom-exdr-550"

output_folder = '/nfs/home/haiphung106/HAINet/output/' + folder + '/inference/cod10k_test/cod10k_fps_550/'
# output_folder = '/nfs/home/haiphung106/HAINet/output/' + folder + '/inference/nc4k_test/nc4k_heats/'

os.makedirs(output_folder, exist_ok=True)
config_file = "/nfs/home/haiphung106/HAINet/configs/CIS-R50.yaml"
weights = "/nfs/home/haiphung106/HAINet/output/" + folder + "/model_0094999.pth"

# Set up configuration
cfg = setup_cfg(config_file, weights)

# Predict and save segmentation maps
predict_instance_segmentation(input_folder, output_folder, cfg)
# predict_heat_map(input_folder, output_folder, cfg)

# import torch
# from fvcore.nn import FlopCountAnalysis, parameter_count

# # Define your model
# model = DefaultPredictor(cfg).model

# # Count parameters
# params = parameter_count(model)
# print(f"Total parameters: {params[''] / 1e6} M")

# # Estimate FLOPs
# input_tensor = torch.randn(1, 3, 1024, 1024)  # Adjust input size as necessary
# flops = FlopCountAnalysis(model, input_tensor)
# print(f"Total FLOPs: {flops.total() / 1e9} GFLOPs")