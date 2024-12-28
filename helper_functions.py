import os
import cv2
from enum import Enum
import torchvision.utils as vutils
import sys

class NoiseType(Enum):
    RANDOM = "random"
    GAUSSIAN = "gaussian"

class GradientDirection(Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    CENTER_TO_SIDES = "center_to_sides"


def return_powder_bounds(img, powder_height):
        
    # generic function to return the boundaries of the image region
    img_height, img_width, _ = img.shape

    top_left = (0, img_height - powder_height)
    bottom_right = (img_width , img_height)

    return top_left, bottom_right

def save_image(img, path):
    """Save the image to the specified path in PNG format."""
    cv2.imwrite(path, img)

def save_images(images, folder, prefix):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, img in enumerate(images):
        vutils.save_image(img, f"{folder}/{prefix}_{i}.png")


    
def progress_bar(iteration, total, prefix='', suffix='', decimals=4, length=1, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()