import json
import os
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def visualize_instance_masks(coco_data, image_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    coco_instance = COCO(coco_data)
    image_ids = coco_instance.getImgIds()

    for img_id in image_ids:
        img_info = coco_instance.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info['file_name'])
        image = cv2.imread(img_path)

        ann_ids = coco_instance.getAnnIds(imgIds=img_id)
        annotations = coco_instance.loadAnns(ann_ids)

        masks = []
        for ann in annotations:
            mask = coco_instance.annToMask(ann)
            masks.append(mask)

        result = visualize_instance_segmentation(image, masks)

        # Save the visualization
        save_path = os.path.join(save_folder, f"visualization_{img_info['file_name']}")
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def visualize_instance_segmentation(image, masks):
    # Convert the image to RGB if it's in BGR format
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blank image with the same shape as the original image
    overlay = np.zeros_like(image)

    # Assign different colors to different instances
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    # Overlay masks on the blank image
    for i, mask in enumerate(masks):
        overlay[mask > 0] = colors[i]

    # Blend the original image and the overlay using alpha blending
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result

if __name__ == "__main__":
    # Specify the paths to the JSON file, image folder, and save folder
    json_path = "./output/hainet-0313-meanshift/inference/cod10k_test/coco_instances_results.json"
    image_folder = "./pool/COD10K/Test_Image_CAM/"
    save_folder = './output/test/'

    # Load COCO annotations and visualize masks
    coco_data = load_coco_annotations(json_path)
    visualize_instance_masks(json_path, image_folder, save_folder)

