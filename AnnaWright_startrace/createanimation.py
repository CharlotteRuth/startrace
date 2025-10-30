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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create animations (GIF or MP4) from a folder of PNG images.",
        epilog="""
Examples:
  # Create MP4 with padding (recommended for varying image sizes)
  python createanimation.py /path/to/images output.mp4 --fps 6
  
  # Create GIF without padding (assumes uniform image sizes)
  python createanimation.py /path/to/images output.gif --fps 12 --simple
  
  # Use default fps (6 for padding, 12 for simple)
  python createanimation.py /path/to/images animation.mp4

Output:
  The script will create an animation file at the specified output path.
  Supported formats: .gif, .mp4
  
  With padding mode (default):
    - Scans all images to find maximum dimensions
    - Pads smaller images with black borders
    - Prevents distortion from varying sizes
  
  With simple mode (--simple flag):
    - Assumes all images have the same dimensions
    - Faster processing, no padding
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input_folder',
        help="Path to the folder containing PNG images"
    )
    parser.add_argument(
        'output_file',
        help="Path for the output animation file (.gif or .mp4)"
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help="Frames per second for the output animation (default: 6 for padding mode, 12 for simple mode)"
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help="Use simple mode without padding (assumes uniform image sizes)"
    )
    
    args = parser.parse_args()
    
    # Set default fps based on mode
    if args.fps is None:
        fps = 12 if args.simple else 6
    else:
        fps = args.fps
    
    # Validate input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        exit(1)
    
    # Validate output file extension
    _, ext = os.path.splitext(args.output_file)
    if ext.lower() not in ['.gif', '.mp4']:
        print(f"Warning: Output file extension '{ext}' may not be supported. Recommended: .gif or .mp4")
    
    # Create the animation
    if args.simple:
        print(f"Creating animation in simple mode (fps={fps})...")
        create_animation_simple(args.input_folder, args.output_file, fps=fps)
    else:
        print(f"Creating animation with padding (fps={fps})...")
        create_animation_with_padding(args.input_folder, args.output_file, fps=fps)