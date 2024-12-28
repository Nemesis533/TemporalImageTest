from ImageGenerator import ImageGeneratorClass
from helper_functions import NoiseType, GradientDirection
import helper_functions as hf

image_save_path = "./image_tests/"
image_base_name = "pfs_"
powder_height = 128
gray_value = 180
image_width = 512
image_height = 256

# Main
if __name__ == "__main__":

    img_gen = ImageGeneratorClass(powder_height)
    # Create a white image
    image = img_gen.generate_base_image(256,512)

    # Add a powder region  
    image = img_gen.add_powder_region(image,gray_value)

    # Add noise to the powder region
    image = img_gen.add_noise(image, 128, 0,powder_height, NoiseType.RANDOM, 0.3)
    image = img_gen.add_noise(image, 128, 0,powder_height, NoiseType.GAUSSIAN,0.3)

    # Add noise to top region
    image = img_gen.add_noise(image, 80, powder_height,image_height , NoiseType.RANDOM, 0.1)
    image = img_gen.add_noise(image, 80, powder_height,image_height , NoiseType.GAUSSIAN,0.1)

    hf.save_image(image, f"{image_save_path}{image_base_name}_nograd.png")

    # Add a gradient to the powder region
    image = img_gen.add_horizontal_gradient(image,GradientDirection.LEFT_TO_RIGHT,80)

    hf.save_image(image, f"{image_save_path}{image_base_name}_1grad.png")

    #add a gradient to the clear region
    image = img_gen.add_horizontal_gradient(image,GradientDirection.LEFT_TO_RIGHT,2)

    hf.save_image(image, f"{image_save_path}{image_base_name}.png")
