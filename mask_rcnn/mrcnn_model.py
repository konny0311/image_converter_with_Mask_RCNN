import os
import sys
import random
import numpy as np
import cv2
import colorsys
import time
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
if 'mask_rcnn' in os.getcwd():
    sys.path.append(os.path.join(os.getcwd(), 'samples', 'coco'))  # To find local version
else:
    sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn', 'samples', 'coco'))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = 'logs'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('mask_rcnn', 'weights', 'mask_rcnn_coco.h5')
OUTPUT_PATH = os.path.join('..', 'output')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRcnnModel():

    def __init__(self, weights_path=None, class_names=None):
        self.config = InferenceConfig()
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        if weights_path is None:
            self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        else:
            self.model.load_weights(weights_path, by_name=True)
        
        if class_names is None:
            self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                        'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear', 'hair drier', 'toothbrush']
        else:
            self.class_names = class_names

    def _apply_mask(self, img, mask, color=None, alpha=0.5):
        # cv2.imwrite('../tmp.jpg', mask)
        copied = img.copy()
        for c in range(3):
            if color is None:
                copied[:, :, c] = np.where(mask == 1,
                                        255, 0) # create binary mask
            else:
                copied[:, :, c] = np.where(mask == 1,
                                    img[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    img[:, :, c])

        return copied

    def _random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def detect(self, img):
        """
        img: image object as numpy.ndarray with 3 channels.
        """
        result = self.model.detect([img])[0]

        return result            


""" Reference to be deleted later
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

"""

if __name__ == '__main__':

    mrcnn_model = MaskRcnnModel()
    img = cv2.imread('../splash_trial2.jpg')
    target = mrcnn_model.detect(img)
