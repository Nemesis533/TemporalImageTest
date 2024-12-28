import cv2
import numpy as np
import helper_functions as hf
from helper_functions import NoiseType, GradientDirection
 
class ImageGeneratorClass:

    def __init__(self,powder_height):
        self.powder_height = powder_height
        pass

    def generate_base_image(self, img_height, img_width):

        #first generate a white image
        img = np.zeros([img_height,img_width,3],dtype=np.uint8)
        img.fill(255) # or img[:] = 255
        return img
    
    def add_powder_region(self,img, grey_value):
        #adds a region that represents the nominal area of powder

        top_left, bottom_right = hf. return_powder_bounds(img,self.powder_height) 

        cv2.rectangle(img,top_left, bottom_right, (grey_value, grey_value, grey_value),-1)

        return img
    
    def add_noise(self, img, max_intensity, min_y, max_y, noise_type: NoiseType, density: float = 1.0):
        # Adds random noise to the image with control over the noise distribution and density

        # Validate density
        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1.")

        # Get the region size
        region_height = max_y - min_y
        region_width = img.shape[1]

        # Generate a mask based on the density
        num_pixels = region_height * region_width
        num_noisy_pixels = int(num_pixels * density)  # Calculate number of noisy pixels based on density

        # Randomly select the indices of pixels to apply noise
        noisy_pixel_indices = np.random.choice(num_pixels, num_noisy_pixels, replace=False)

        # Get the 2D coordinates (row, col) of the noisy pixels
        rows = noisy_pixel_indices // region_width + min_y  # Calculate the row indices
        cols = noisy_pixel_indices % region_width         # Calculate the column indices

        if noise_type == NoiseType.GAUSSIAN:
            mean = 0
            stddev = max_intensity / 3  # Standard deviation for the noise
            noise = np.random.normal(mean, stddev, (num_noisy_pixels, 3)).astype(np.uint8)
            noise = np.clip(noise, 0, max_intensity)

        elif noise_type == NoiseType.RANDOM:
            noise = np.random.randint(0, max_intensity + 1, (num_noisy_pixels, 1), dtype=np.uint8)

        # Apply the generated noise to the selected pixels
        for i in range(num_noisy_pixels):
            img[rows[i], cols[i], :] = noise[i]

        return img
    
    def add_horizontal_gradient(self, img, direction : GradientDirection, intensity):

        #adds a horizontal gradient to the image

        img_height, img_width, _ = img.shape

        if direction == GradientDirection.LEFT_TO_RIGHT:
            gradient = np.linspace(0, intensity, img_width, dtype=np.uint8).reshape(1, -1)
            gradient_img = np.repeat(gradient, img_height, axis=0)
        elif direction == GradientDirection.RIGHT_TO_LEFT:
            gradient = np.linspace(intensity, 0, img_width, dtype=np.uint8).reshape(1, -1)
            gradient_img = np.repeat(gradient, img_height, axis=0)
        elif direction == GradientDirection.CENTER_TO_SIDES:
            center = img_width // 2
            left_gradient = np.linspace(intensity, 0, center, dtype=np.uint8)
            right_gradient = np.linspace(0, intensity, img_width - center, dtype=np.uint8)
            gradient = np.concatenate([left_gradient, right_gradient]).reshape(1, -1)
            gradient_img = np.repeat(gradient, img_height, axis=0)
        else:
            raise ValueError("Incorrect gradient options. Choose a valid GradientDirection.")

        for column in range(3):  # For each color channel
            img[:, :, column] = np.clip(img[:, :, column] + gradient_img, 0, 255)

        return img

