
class config:

    #########
    # POSE CONFIGS:
    #########

    # Number of keypoints
    #NUM_KP = 17
    NUM_KP = 23

    # List of keypoint names
    KEYPOINTS = [
        "nose",         # 0
        # "neck",       
        "Rshoulder",    # 1
        "Relbow",       # 2
        "Rwrist",       # 3
        "Lshoulder",    # 4
        "Lelbow",       # 5
        "Lwrist",       # 6
        "Rhip",         # 7
        "Rknee",        # 8
        "Rankle",       # 9
        "Lhip",         # 10
        "Lknee",        # 11
        "Lankle",       # 12
        "Reye",         # 13
        "Leye",         # 14
        "Rear",         # 15
        "Lear",         # 16
        "Rheel",        # 17 - extra starts here
        "Rbigtoe",      # 18
        "Rlittletoe",   # 19
        "Lheel",        # 20
        "Lbigtoe",      # 21
        "Llittletoe"    # 22
    ]

    # Indices of right and left keypoints (for flipping in augmentation)
    #RIGHT_KP = [1, 2, 3,  7,  8,  9, 13, 15]
    RIGHT_KP = [1, 2, 3,  7,  8,  9, 13, 15, 17, 18, 19]
    #LEFT_KP =  [4, 5, 6, 10, 11, 12, 14, 16]
    LEFT_KP =  [4, 5, 6, 10, 11, 12, 14, 16, 20, 21, 22]

    # List of edges as tuples of indices into the KEYPOINTS array
    # (Each edge will be used twice in the mid-range offsets; once in each direction)
    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3),
        (9,17), #extra starts here
        (9,18),
        (9,19),
        (12,20),
        (12,21),
        (12,22)
    ]

    NUM_EDGES = len(EDGES)

    #########
    # PRE- and POST-PROCESSING CONFIGS:
    #########

    # Radius of the discs around the keypoints. Used for computing the ground truth
    # and computing the losses. (Recommended to be a multiple of the output stride.)
    KP_RADIUS = 16

    # The threshold for extracting keypoints from hough maps.
    PEAK_THRESH = 0.0004

    # Pixel distance threshold for whether to begin a new skeleton instance
    # (If another skeleton already has this keypoint within the threshold, it is discarded.)
    NMS_THRESH = 16

    # The metric threshold for assigning a pixel to a given instance mask 
    INSTANCE_SEG_THRESH = 0.25

    #########
    # TRAINING CONFIGS:
    #########

    # Input shape for training images (By convention s*n+1 for some integer n and s=output_stride)
    #IMAGE_SHAPE = (401, 401, 3)
    IMAGE_SHAPE = (65, 65, 3)

    # Output stride of the base network (resnet101 or resnet152 in the paper)
    # [Any convolutional stride in the original network which would reduce the 
    # output stride further is replaced with a corresponding dilation rate.]
    OUTPUT_STRIDE = 8

    # Weights for the losses applied to the keypoint maps ('heatmap'), the binary segmentation map ('seg'),
    # and the short-, mid-, and long-range offsets.
    #LOSS_WEIGHTS = {
    #    'heatmap': 4,
    #    'seg': 2,
    #    'short': 1,
    #    'mid': 0.25,
    #    'long': 0.125
    #}
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'short': 0.25,
        'mid': 0.125,
    }

    # The filepath for the training dataset
    H5_DATASET = './dataset/coco2017_personlab_train.h5'

    # Whether to keep the batchnorm weights frozen.
    BATCH_NORM_FROZEN = False

    # Number of GPUs to distribute across
    NUM_GPUS = 1

    # The total batch size will be (NUM_GPUS * BATCH_SIZE_PER_GPU)
    BATCH_SIZE_PER_GPU = 1

    # Whether to use Polyak weight averaging as mentioned in the paper
    POLYAK = False

    # Optional model weights filepath to use as initialization for the weights
    LOAD_MODEL_PATH = None
    #LOAD_MODEL_PATH = './personlab_model.h5'

    # Where to save the model.
    SAVE_MODEL_PATH = './models/foot_posenet_101.h5'

    # Epochs
    NUM_EPOCHS = 3


class TransformationParams:

    target_dist = 0.8
    scale_prob = 1.
    scale_min = 0.8
    scale_max = 2.0
    max_rotate_degree = 25.
    center_perterb_max = 0.
    flip_prob = 0.5
