import torch
import cv2
import numpy as np
import argparse


def process_image(file_num, model_path='./checkpoints/saved_pix2pix.pt'):
    """Process an input image using a trained generator model.

    Args:
        file_num (int): Number identifier for input/output files
        model_path (str): Path to trained generator model
    """
    # Load trained generator model
    generator = torch.load(model_path).cuda()
    generator.eval()  # Set to evaluation mode

    # Define file paths
    input_file = f'./fig{file_num}_resize.jpg'
    middle_file = f'./fig{file_num}_middle.jpg'
    output_file = f'./fig{file_num}_output.jpg'

    # Readprocess input image
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    # Edge detection
    _, edges = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(middle_file, edges)  # Save intermediate result
    edges = cv2.Canny(edges, threshold1=100, threshold2=200)

    # Prepare input tensor
    image = np.stack([image, edges]) / 255.0  # Normalize and stack channels
    image_tensor = torch.FloatTensor(image).unsqueeze(0).cuda()  # Add batch dim

    # Generate output
    with torch.no_grad():
        output = generator(image_tensor)

    # Post-process output
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)  # Remove batch and channel dims
    output = (output * 255).astype(np.uint8)  # Scale to 0-255

    # Save result
    cv2.imwrite(output_file, output)
    print(f"Processing complete. Output saved to {output_file}")


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Image processing using GAN')
    parser.add_argument(
        'file_num',
        type=int,
        help='File number identifier (e.g. 16 for fig16_resize.jpg)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./checkpoints/saved_pix2pix.pt',
        help='Path to generator model'
    )

    # Parse arguments and run processing
    args = parser.parse_args()
    process_image(args.file_num, args.model)