import cv2
import numpy as np
import argparse
from utils import *


def reProcess(fig_num):
    """Process and rebuild molecular structure from output image"""
    # Load and preprocess images
    img_name = f'./fig{fig_num}_output.jpg'
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image.shape

    # Load middle reference image
    middle_image = cv2.imread(f'fig{fig_num}_middle.jpg', cv2.IMREAD_GRAYSCALE)

    # Detect carbon atom positions
    carbon_positions = calculate_carbon_positions(image)
    print("Detected carbon positions:", carbon_positions)

    # Find hexagonal patterns in the carbon positions
    hexagon_points_to_center, mean_edge_len = find_hexagons(
        carbon_positions,
        edge_min=11,
        edge_max=19,
        angle_tolerance=np.radians(15)
    )
    print("Mean edge length:", mean_edge_len)

    # Prepare vertex and center data
    all_centers = [[int(x), int(y)] for x, y in hexagon_points_to_center.values()]
    all_vertices = list(hexagon_points_to_center.keys())
    all_vertices_backup = all_vertices.copy()

    # Process each hexagon to refine edges
    for index in range(len(all_centers)):
        traverse_hexagon_edges(
            all_vertices[index],
            middle_image,
            all_vertices,
            all_centers,
            index,
            mean_edge_len=mean_edge_len,
            radius=mean_edge_len,
            ratio=0.99,
            debug=True
        )

    # Flatten and merge close vertices
    all_vertices = [item for sublist in all_vertices for item in sublist]
    print("Vertex count before merging:", len(all_vertices))

    all_vertices = merge_close_points(all_vertices, 8)
    cv2.imwrite('./middle1.png', process_locs(transform_coor(all_vertices, image_size), image_size))

    # Further processing commented out but available if needed
    # all_vertices = merge_close_points(all_vertices, 8)

    # Save final rebuilt structure
    cv2.imwrite(
        f'./rebuild_fig{fig_num}.png',
        process_locs(transform_coor(all_vertices, image_size), image_size)
    )

    # Save backup vertices for comparison
    all_vertices_backup = [item for sublist in all_vertices_backup for item in sublist]
    cv2.imwrite(
        './middle3.png',
        process_locs(transform_coor(all_vertices_backup, image_size), image_size)
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Process molecular structure images')
    parser.add_argument(
        'fig_num',
        type=int,
        help='Figure number to process (e.g., 9)'
    )
    args = parser.parse_args()

    # Run processing with the provided figure number
    reProcess(args.fig_num)
