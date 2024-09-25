import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

COD10K_ROOT = './pool/COD10K/'
ANN_ROOT = os.path.join(COD10K_ROOT, 'Train_instance_gt_CAM')
TRAIN_PATH = os.path.join(COD10K_ROOT, 'Train_Image_CAM')
TEST_PATH = os.path.join(COD10K_ROOT, 'Test_Image_CAM')
TRAIN_JSON = os.path.join(COD10K_ROOT, 'train_instance.json')
TEST_JSON = os.path.join(COD10K_ROOT, 'test2026.json')

NC4K_ROOT = './pool/NC4K/'
NC4K_PATH = os.path.join(NC4K_ROOT, 'test/image')
NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')

coco2017_ROOT = './pool/coco/'
coco2017_PATH = os.path.join(coco2017_ROOT, 'val2017')
coco2017_JSON = os.path.join(coco2017_ROOT, 'coco-2017-dataset-metadata.json')


CLASS_NAMES = ["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "cod10k_train": (TRAIN_PATH, TRAIN_JSON),
    "cod10k_test": (TEST_PATH, TEST_JSON),
    "nc4k_test": (NC4K_PATH, NC4K_JSON),
    "coco_2017_val": (coco2017_PATH, coco2017_JSON)
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")