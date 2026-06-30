import os
import torch

# --- Dataset Paths ---
ARCHIVE_ROOT     = "/Users/harundolcan/Downloads/archive (1)"
IMAGES_DIR       = os.path.join(ARCHIVE_ROOT, "images")
ANNOTATIONS_DIR  = os.path.join(ARCHIVE_ROOT, "annotations")

YOLO_ROOT        = "/Users/harundolcan/Downloads/ppe detection.yolov8"
USE_YOLO_DATASET = True

# --- Classes ---
# 0 = background (reserved by Faster R-CNN)
# 1 = helmet  (helmet ON the head)
# 2 = head    (bare head = No Helmet)
NUM_CLASSES    = 3
CLASS_NAMES    = ["background", "helmet", "head"]
CLASS_MAP      = {"helmet": 1, "head": 2}
YOLO_CLASS_MAP = {0: 1, 1: 2}   # YOLO 0=Helmet, 1=No_Helmet

# --- Dataset Split ---
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

# --- Training ---
BATCH_SIZE          = 4
NUM_EPOCHS          = 10
LEARNING_RATE       = 0.005
MOMENTUM            = 0.9
WEIGHT_DECAY        = 0.0005
LR_STEP_SIZE        = 10
LR_GAMMA            = 0.1
EVAL_EVERY_N_EPOCHS = 2

# --- Model ---
# weights=None → NO COCO detection weights (professor requirement)
# ImageNet backbone is allowed — it is NOT COCO pre-training
USE_IMAGENET_BACKBONE = True

# --- Paths ---
CHECKPOINT_DIR = "./checkpoints"
SPLITS_FILE    = "./data_splits.json"
RESULTS_FILE   = "./results.json"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")
