from detectron2.config import CfgNode as CN


def add_net_config(cfg):
    """
    Add config for Mask2Camouflage.
    """

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False

    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1


    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # NET model config
    cfg.MODEL.NET = CN()

    # loss
    cfg.MODEL.NET.DEEP_SUPERVISION = True
    # cfg.MODEL.NET.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.NET.CLASS_WEIGHT = 1.0
    cfg.MODEL.NET.DICE_WEIGHT = 1.0
    cfg.MODEL.NET.MASK_WEIGHT = 20.0
    cfg.MODEL.NET.DEC_LAYERS = 6
    cfg.MODEL.NET.NUM_OBJECT_QUERIES = 10
    cfg.MODEL.NET.SIZE_DIVISIBILITY = 32


    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn  configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    
    
    
    
    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False


    