import os
import sys
from PIL import Image
from torchvision import transforms  



# execute: python create.py IMG_1.jpg




def data_augmentation(image):

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
        transforms.RandomRotation(degrees=30),   # Rotate randomly by up to 20 degrees
        transforms.RandomResizedCrop(size=(160, 160), scale=(0.8, 1.0)),  # Randomly crop and resize to 160x160
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.2),  # Adjust color jitters
    ])

    return data_transforms(image)

if __name__ == "__main__":
    
    # Get the image path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python create.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # Create the output directory (if it doesn't exist)
    output_directory = 'augmented_images'
    os.makedirs(output_directory, exist_ok=True, mode=0o755)

    # Generate and save 20 augmented images
    for i in range(20):
        augmented_image = data_augmentation(image.copy())  # Use a copy to avoid modifying the original image
        augmented_image.save(os.path.join(output_directory, f'augmented_image_{i}.jpg'))

    print(f'Successfully generated and saved 20 images to {output_directory}')












