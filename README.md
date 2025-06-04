# StringArt

Python String Art Generator that transforms an input image into a captivating string art representation. It uses image processing techniques and a greedy algorithm to simulate strings stretched between pins, recreating the image's outlines.


## Configuration Parameters

Key parameters can be adjusted in the `if __name__ == "__main__":` block of `string_art_generator.py`:

*   `IMAGE_PATH`: Path to your input image (e.g., `'my_picture.jpg'`).
*   `SHOW_INTERMEDIATE_PLOTS`: `True` or `False`. If `True`, displays images at various processing stages.
*   `CANNY_SIGMA`: Sigma value for the Canny edge detection. Higher values mean less sensitivity to noise and fewer, smoother edges.
*   `NUM_NAILS`: Total number of nails to place around the circumference of the image.
*   `NAIL_PADDING`: Padding (in pixels) from the image edge to the circle where nails are placed.
*   `NAIL_VISUAL_RADIUS`: Radius (in pixels) for visually plotting the nails in the intermediate "Canvas with Nails" plot. Does not affect line drawing.
*   `TARGET_BLUR_SIGMA`: Sigma for the Gaussian blur applied to the inverted Canny edges. This helps create a softer target for the string art algorithm.
*   `NUM_LINES`: The maximum number of lines (strings) the algorithm will attempt to draw.
*   `LINE_STRENGTH`: A value typically between 0.0 and 1.0 that determines how much each line darkens the canvas. Smaller values result in fainter lines and potentially more lines being drawn.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
