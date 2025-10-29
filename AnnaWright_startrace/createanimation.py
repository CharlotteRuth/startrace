import imageio.v2 as imageio
import numpy as np
import glob
import os
import re
import tqdm.auto as tqdm
import argparse

def create_animation_with_padding(input_folder, output_file, fps=6):
    """
    Creates a GIF or MP4 from a folder of PNG images with varying sizes.

    It finds the largest image dimensions and pads smaller images with black
    borders to create uniformly-sized frames, preventing distortion.
    
    Args:
        input_folder (str): Path to the folder containing the PNG images.
        output_file (str): Path for the output animation file. The extension 
                           (.gif or .mp4) determines the format.
        fps (int): Frames per second for the output animation.
    """
    search_path = os.path.join(input_folder, '*.png')
    filenames = glob.glob(search_path)

    if not filenames:
        print(f"Error: No PNG files found in '{input_folder}'")
        return

    # Sort files naturally to handle numbers like 'plot_2.png' before 'plot_10.png'
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    filenames.sort(key=natural_sort_key)

    # --- Step 1: Scan all images to find the maximum dimensions ---
    print("Scanning images to determine maximum dimensions...")
    max_height = 0
    max_width = 0
    
    # Also determine the color channel configuration (RGB vs RGBA) from the first image
    first_image = imageio.imread(filenames[0])
    print(f"First image shape: {first_image.shape}")
    channels = first_image.shape[2] if len(first_image.shape) == 3 else 1
    dtype = first_image.dtype

    for filename in filenames:
        img = imageio.imread(filename)
        if len(img.shape) == 3: # Ensure it's a color image
            h, w, _ = img.shape
            max_height = max(max_height, h)
            max_width = max(max_width, w)

    print(f"All frames will be resized to {max_width}x{max_height} pixels.")

    # --- Step 2: Create the animation with padded frames ---
    print(f"Creating animation at '{output_file}'...")
    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in tqdm.tqdm(filenames, desc="Processing frames"):
            # Read the original image
            img = imageio.imread(filename)
            
            # Create a new black canvas with the target dimensions
            padded_frame = np.zeros((max_height, max_width, channels), dtype=dtype)
            
            # Calculate offsets to center the image
            h, w, _ = img.shape
            y_offset = (max_height - h) // 2
            x_offset = (max_width - w) // 2
            
            # Paste the original image onto the canvas
            padded_frame[y_offset:y_offset+h, x_offset:x_offset+w, :] = img
            
            # Append the padded frame to the video
            writer.append_data(padded_frame)

    print("Animation created successfully!")

def create_animation_simple(input_folder, output_file, fps=12):
    """
    Creates a GIF or MP4 from a folder of PNG images.

    The function assumes filenames can be sorted naturally (e.g., frame_1.png, frame_2.png, ... frame_10.png).

    Args:
        input_folder (str): Path to the folder containing the PNG images.
        output_file (str): Path for the output animation file. The extension (.gif or .mp4) determines the format.
        fps (int): Frames per second for the output animation.
    """
    search_path = os.path.join(input_folder, '*.png')
    filenames = glob.glob(search_path)

    if not filenames:
        print(f"Error: No PNG files found in the directory: {input_folder}")
        return

    filenames.sort(key=lambda x: str.split(str.split(x, '.')[-2], '_')[-1])  # Sort by the numeric part of the filename

    print(f"Found {len(filenames)} images. Creating animation at {output_file}...")

    # Create the animation
    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("Animation created successfully!")
    
# # --- Example Usage ---
# # Replace this with the actual path to your saved plots
# image_folder = '/home/selvani/MAP/pynbody/stellarhalo_trace_aw/merge_plots/3' 

# # Define the output file path (can be .gif or .mp4)
# output_path = os.path.join(image_folder, 'merger_animation.mp4')

# # Create the animation
# create_animation_with_padding(image_folder, output_path, fps=5)

if __name__ == "__main__":
    raise NotImplementedError("This module is intended to be imported and used by other scripts.")