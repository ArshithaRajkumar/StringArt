# ---------------------------------------------------------------------------- #
#                             String Art Generator                             #
# ---------------------------------------------------------------------------- #
# Author: [Arshitha Rajkumar & Pushap Raj]
# Description: This script takes an input image, processes it, and generates
#              a string art representation based on Canny edge detection.
# ---------------------------------------------------------------------------- #

# %% ---------------------------- IMPORTS ----------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
from skimage.draw import line_aa,disk
from skimage.filters import gaussian

# %% ---------------------------- FUNCTIONS --------------------------------- #

def rgb_to_grayscale(image_rgb):
    """Converts an RGB image to grayscale using standard luminance weights."""
    if image_rgb.ndim == 2: 
        return image_rgb
    if image_rgb.shape[2] == 4: 
        image_rgb = image_rgb[:,:,:3] 

    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b #RGBA Conversion
    return grayscale / 255.0  

def largest_square(image_gray):
    """Crops the largest possible square from the center of the image."""
    height, width = image_gray.shape[:2]
    if height <= width:
        short_edge = height
        long_edge_half = width // 2
        short_edge_half = short_edge // 2
        return image_gray[:, long_edge_half - short_edge_half : long_edge_half + short_edge_half]
    else:
        short_edge = width
        long_edge_half = height // 2
        short_edge_half = short_edge // 2
        return image_gray[long_edge_half - short_edge_half : long_edge_half + short_edge_half, :]

def create_circle_nail_positions(image_shape, num_nails, padding=10):
    """
    Generates coordinates for nails arranged in a circle.
    
    Args:
        image_shape (tuple): Shape of the image (height, width).
        num_nails (int): Number of nails to place around the circle.
        padding (int): Padding from the edge of the image to the nail circle.

    Returns:
        list: A list of (y, x) coordinates for each nail.
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 2 - padding

    nails = []
    for i in range(num_nails):
        theta = 2 * np.pi * i / num_nails
        y = int(center_y + radius * np.sin(theta))
        x = int(center_x + radius * np.cos(theta))
        y = np.clip(y, 0, height -1)
        x = np.clip(x, 0, width - 1)
        nails.append((y, x))
    return nails

def get_line_pixels_and_overlay(from_pos, to_pos, line_strength, current_canvas):
    """
    Calculates pixels for an anti-aliased line and overlays it onto a copy of the canvas.
    
    Args:
        from_pos (tuple): Starting (row, col) of the line.
        to_pos (tuple): Ending (row, col) of the line.
        line_strength (float): Factor by which to darken pixels along the line.
        current_canvas (np.array): The canvas on which the line is drawn.

    Returns:
        tuple: (overlayed_canvas, valid_rows, valid_cols)
               overlayed_canvas is a new canvas with the line drawn.
               valid_rows, valid_cols are the coordinates of the line pixels.
    """
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    
    overlayed_canvas = current_canvas.copy()
    
    valid_mask = (rr >= 0) & (rr < overlayed_canvas.shape[0]) & \
                 (cc >= 0) & (cc < overlayed_canvas.shape[1])
    rr_valid, cc_valid, val_valid = rr[valid_mask], cc[valid_mask], val[valid_mask]
    
    overlayed_canvas[rr_valid, cc_valid] = np.clip(
        overlayed_canvas[rr_valid, cc_valid] - line_strength * val_valid, 0, 1
    )
    return overlayed_canvas, rr_valid, cc_valid

def find_best_next_nail(current_nail_pos, current_nail_idx, all_nails, 
                        current_art_canvas, target_image, line_strength):
    """
    Finds the best next nail to connect to from the current nail.
    "Best" is defined as the line that maximizes the reduction in squared error
    between the current art canvas and the target image.
    """
    best_improvement = 0.0  #positive improvements
    best_next_nail_pos = None
    best_next_nail_idx = None

    for idx, next_nail_candidate_pos in enumerate(all_nails):
        if idx == current_nail_idx: 
            continue

        line_overlay_pixels, rr, cc = get_line_pixels_and_overlay(
            current_nail_pos, next_nail_candidate_pos, line_strength, current_art_canvas
        )

        if len(rr) == 0: 
            continue
        
        # Error = (canvas_pixel - target_pixel)^2
        error_before_line_sq = (current_art_canvas[rr, cc] - target_image[rr, cc]) ** 2
        error_after_line_sq = (line_overlay_pixels[rr, cc] - target_image[rr, cc]) ** 2

        improvement = np.sum(error_before_line_sq - error_after_line_sq)

        if improvement > best_improvement:
            best_improvement = improvement
            best_next_nail_pos = next_nail_candidate_pos
            best_next_nail_idx = idx
            
    return best_next_nail_idx, best_next_nail_pos, best_improvement

def generate_string_art(target_image, nail_coords, num_lines, line_strength_param):
    """
    Generates the string art by iteratively finding and drawing the best lines.
    """
    string_art_canvas = np.ones_like(target_image)  #white canvas
    current_nail_idx = 0  
    line_path_indices = [current_nail_idx]  # Stores the sequence of nail indices

    print(f"Starting string art generation: {num_lines} lines, strength: {line_strength_param}")
    
    for i in range(num_lines):
        current_nail_pos = nail_coords[current_nail_idx]
        
        best_idx, best_pos, improvement = find_best_next_nail(
            current_nail_pos,
            current_nail_idx,
            nail_coords,
            string_art_canvas,
            target_image,
            line_strength=line_strength_param
        )

        if best_idx is None:
            print(f"Stopping at line {i+1}/{num_lines}: No further positive improvement found.")
            break
        
        string_art_canvas, _, _ = get_line_pixels_and_overlay(
            current_nail_pos, best_pos, line_strength_param, string_art_canvas
        )

        line_path_indices.append(best_idx)
        current_nail_idx = best_idx
        
        if (i + 1) % 100 == 0 or (i + 1) == num_lines:
            print(f"Line {i+1}/{num_lines} drawn. Last improvement: {improvement:.3f}")
    else: 
        if num_lines > 0 :
             print(f"Completed all {num_lines} lines.")
             
    return string_art_canvas, line_path_indices

def plot_image(image_data, title, cmap='gray', axis_off=True):
    """Helper function for plotting images."""
    plt.imshow(image_data, cmap=cmap)
    plt.title(title)
    if axis_off:
        plt.axis('off')
    plt.show()

# %% ---------------------------- MAIN SCRIPT ------------------------------- #
if __name__ == "__main__":

    # --- Configuration ---
    IMAGE_PATH = 'Ganesha - Copy.png' 
    SHOW_INTERMEDIATE_PLOTS = True   

    # Canny edge detection parameters
    CANNY_SIGMA = 1.0

    # Nail configuration
    NUM_NAILS = 200          
    NAIL_PADDING = 10       
    NAIL_VISUAL_RADIUS = 2  

    # Target image processing
    TARGET_BLUR_SIGMA = 1.5 # Sigma for Gaussian blur of the inverted Canny edges

    # String art generation parameters
    NUM_LINES = 3000        
    LINE_STRENGTH = 0.07   
    
    # --- 1. Load and Preprocess Image ---
    print(f"Loading image: {IMAGE_PATH}")
    original_image_rgb = io.imread(IMAGE_PATH)
    if SHOW_INTERMEDIATE_PLOTS:
        plot_image(original_image_rgb, "Original Image", cmap=None)

    grayscale_image = rgb_to_grayscale(original_image_rgb)
    if SHOW_INTERMEDIATE_PLOTS:
        plot_image(grayscale_image, "Grayscale Image")

    square_grayscale_image = largest_square(grayscale_image)
    if SHOW_INTERMEDIATE_PLOTS:
        plot_image(square_grayscale_image, "Largest Square Cropped Image")

    # --- 2. Edge Detection ---
    print("Performing Canny edge detection...")
    canny_edges = feature.canny(square_grayscale_image, sigma=CANNY_SIGMA)
    if SHOW_INTERMEDIATE_PLOTS:
        plot_image(1 - canny_edges, f"Inverted Canny Edges (sigma={CANNY_SIGMA})")

    # --- 3. Prepare Target Image for String Art ---
    print("Preparing target image for string art...")
    base_target_image = 1.0 - canny_edges.astype(np.float32)
    blurred_target = gaussian(base_target_image, sigma=TARGET_BLUR_SIGMA)
    min_val, max_val = blurred_target.min(), blurred_target.max()
    if max_val > min_val:
        string_art_target_image = (blurred_target - min_val) / (max_val - min_val)
    else: 
        string_art_target_image = np.zeros_like(blurred_target) if min_val == 0 else np.ones_like(blurred_target) * blurred_target[0,0]
    
    print(f"  Target image min/max before rescaling: {min_val:.4f}, {max_val:.4f}")
    print(f"  Target image min/max after rescaling: {string_art_target_image.min():.4f}, {string_art_target_image.max():.4f}")
    if SHOW_INTERMEDIATE_PLOTS:
        plot_image(string_art_target_image, f"Final Target Image (Blurred & Rescaled, sigma={TARGET_BLUR_SIGMA})")

    # --- 4. Define Nail Positions ---
    print(f"Creating {NUM_NAILS} nail positions...")
    nail_coordinates = create_circle_nail_positions(
        square_grayscale_image.shape, NUM_NAILS, padding=NAIL_PADDING
    )

    if SHOW_INTERMEDIATE_PLOTS:
        canvas_with_nails = np.ones(square_grayscale_image.shape, dtype=np.float32)
        for r_nail, c_nail in nail_coordinates:
            # --- MODIFICATION FOR THICKER NAILS ---
            rr_nail, cc_nail = disk((r_nail, c_nail), NAIL_VISUAL_RADIUS, shape=canvas_with_nails.shape)
            canvas_with_nails[rr_nail, cc_nail] = 0  # Mark nail area as black
            # --- END MODIFICATION ---
        plot_image(canvas_with_nails, f"Canvas with {NUM_NAILS} Nails")


    # --- 5. Generate String Art ---
    final_string_art, final_pull_order = generate_string_art(
        string_art_target_image, nail_coordinates, NUM_LINES, LINE_STRENGTH
    )

    # --- 6. Display Final Result ---
    plot_image(
        final_string_art,
        f"Final String Art ({len(final_pull_order)-1} lines, str: {LINE_STRENGTH}, nails: {NUM_NAILS})"
    )
    print(f"Number of lines drawn: {len(final_pull_order) -1}")
    print(f"Final art min/max pixel values: {final_string_art.min():.4f}, {final_string_art.max():.4f}")
    
# Optional: Save the image
# final_image_to_save = (final_string_art * 255).astype(np.uint8)
# io.imsave("final_string_art.png", final_image_to_save)
# print("Final string art saved to final_string_art.png")

# Optional: Save the pull order (sequence of nail indices)
# np.savetxt("pull_order.txt", np.array(final_pull_order), fmt='%d')
# print("Pull order saved to pull_order.txt")