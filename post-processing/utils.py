import cv2
import numpy as np
import math
from collections import defaultdict


def scale_points(points, image_width, image_height, scale_factor):
    """Scale points around the image center."""
    center_x = image_width / 2
    center_y = image_height / 2

    scaled_points = []
    for x, y in points:
        offset_x = x - center_x
        offset_y = y - center_y
        scaled_offset_x = offset_x * scale_factor
        scaled_offset_y = offset_y * scale_factor
        scaled_x = center_x + scaled_offset_x
        scaled_y = center_y + scaled_offset_y
        scaled_points.append((int(scaled_x), int(scaled_y)))

    return scaled_points


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_carbon_positions(image):
    """Detect carbon atom positions using image processing."""
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    cv2.imwrite(f'erode.png', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    carbon_positions = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # Centroid x-coordinate
            cy = int(M['m01'] / M['m00'])  # Centroid y-coordinate
            carbon_positions.append((cx, cy))
    return carbon_positions


def resize_and_pad(image, scale_factor):
    """Resize image with padding to maintain original dimensions."""
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = np.zeros((original_height, original_width), dtype=np.uint8)
    start_x = (original_width - new_width) // 2
    start_y = (original_height - new_height) // 2
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image
    return padded_image


def angle_between(v1, v2):
    """Calculate angle between two vectors in radians."""
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return np.arccos(dot_product / (magnitude_v1 * magnitude_v2))


def find_hexagons(carbon_positions, edge_min=11, edge_max=19, angle_tolerance=np.radians(15)):
    """Find hexagonal patterns in carbon positions."""
    candidate_centers = []
    hexagon_points_to_center = {}
    hexagon_edges = []

    for x in range(0, 640):
        for y in range(0, 640):
            hexagon_points = find_hexagon(carbon_positions, [x, y],
                                          edge_min=edge_min, edge_max=edge_max,
                                          angle_tolerance=angle_tolerance)
            if hexagon_points:
                if hexagon_points in hexagon_points_to_center:
                    hexagon_points_to_center[hexagon_points].append([x, y])
                else:
                    hexagon_points_to_center[hexagon_points] = [[x, y]]
                candidate_centers.append([x, y])

                # Calculate hexagon edge lengths
                edges = []
                for i in range(6):
                    current_point = hexagon_points[i]
                    next_point = hexagon_points[(i + 1) % 6]
                    d = calculate_distance(current_point, next_point)
                    edges.append(d)
                hexagon_edges.append(edges)

    # Average centers for duplicate hexagons
    hexagon_points_to_center = {key: list(np.mean(value, axis=0))
                                for key, value in hexagon_points_to_center.items()}

    # Calculate mean edge length
    hexagon_edges = np.array(hexagon_edges) if hexagon_edges else None
    mean_edge_len = np.mean(hexagon_edges) if hexagon_edges is not None else None

    return hexagon_points_to_center, mean_edge_len


def find_hexagon(coords, center, edge_min, edge_max, angle_tolerance):
    """Find a hexagon pattern around a given center point."""
    possible_points = [p for p in coords if p != center]
    hexagon_points = []

    # Filter points within edge length range
    for p in possible_points:
        d = calculate_distance(center, p)
        if edge_min <= d <= edge_max:
            hexagon_points.append(p)

    if len(hexagon_points) < 6:
        return None

    # Sort points by angle for sequential processing
    sorted_points = sorted(hexagon_points,
                           key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

    # Check for valid hexagon pattern
    for i in range(len(sorted_points) - 5):
        candidate_points = sorted_points[i:i + 6]
        valid_hexagon = True

        for j in range(6):
            current_point = candidate_points[j]
            next_point = candidate_points[(j + 1) % 6]
            prev_point = candidate_points[(j - 1) % 6]

            # Calculate vectors and angles
            v1 = np.array([current_point[0] - prev_point[0],
                           current_point[1] - prev_point[1]])
            v2 = np.array([next_point[0] - current_point[0],
                           next_point[1] - current_point[1]])
            angle = angle_between(v1, v2)

            # Validate angle and edge length
            if not (np.radians(60) - angle_tolerance <= angle <= np.radians(60) + angle_tolerance):
                valid_hexagon = False
                break

            d = calculate_distance(current_point, next_point)
            if not (edge_min <= d <= edge_max):
                valid_hexagon = False
                break

        if valid_hexagon:
            return tuple(candidate_points)

    return None


def scale_point(center, point, target_distance=14.5):
    """Scale a point to a specific distance from the center."""
    current_distance = calculate_distance(center, point)
    scaling_factor = target_distance / current_distance
    x = center[0] + scaling_factor * (point[0] - center[0])
    y = center[1] + scaling_factor * (point[1] - center[1])
    return [x, y]


def generate_hexagon(center, candidate_points, edge_length=14.5):
    """Generate a regular hexagon given a center and candidate points."""
    # Find closest point to target edge length
    closest_point = None
    min_diff = float('inf')

    for point in candidate_points:
        d = calculate_distance(point, center)
        diff = abs(d - edge_length)
        if diff < min_diff:
            min_diff = diff
            closest_point = point

    if closest_point is None:
        print("No suitable starting point found.")
        return None

    # Scale point to exact distance
    start_point = scale_point(center, closest_point, target_distance=edge_length)

    # Generate hexagon vertices
    angle = np.arctan2(start_point[1] - center[1], start_point[0] - center[0])
    hexagon_points = []
    for i in range(6):
        current_angle = angle + np.radians(60 * i)
        x = center[0] + edge_length * np.cos(current_angle)
        y = center[1] + edge_length * np.sin(current_angle)
        hexagon_points.append([int(x), int(y)])

    return hexagon_points


def calculate_centroid(vertices):
    """Calculate centroid of a polygon."""
    x_coords = [p[0] for p in vertices]
    y_coords = [p[1] for p in vertices]
    centroid = (sum(x_coords) / len(vertices), sum(y_coords) / len(vertices))
    return centroid


def calculate_angle(centroid, point):
    """Calculate angle between centroid and point."""
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def calculate_midpoint(p1, p2):
    """Calculate midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def calculate_external_normal(p1, p2, centroid):
    """Calculate external normal vector for an edge."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    normal = (-dy, dx)  # Initial normal

    # Determine outward direction
    midpoint = calculate_midpoint(p1, p2)
    vector_to_centroid = (centroid[0] - midpoint[0], centroid[1] - midpoint[1])
    dot_product = normal[0] * vector_to_centroid[0] + normal[1] * vector_to_centroid[1]

    if dot_product > 0:  # If pointing inward, flip
        normal = (-normal[0], -normal[1])

    # Normalize
    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
    return (normal[0] / length, normal[1] / length)


def generate_new_center(p1, p2, edge_length, centroid):
    """Generate center for adjacent hexagon."""
    midpoint = calculate_midpoint(p1, p2)
    normal = calculate_external_normal(p1, p2, centroid)
    move_distance = edge_length * math.sqrt(3) / 2
    hex_center = (midpoint[0] + normal[0] * move_distance,
                  midpoint[1] + normal[1] * move_distance)
    return hex_center


def generate_hexagon_around_edge(p1, p2, edge_length, centroid):
    """Generate hexagon centered around an edge."""
    midpoint = calculate_midpoint(p1, p2)
    normal = calculate_external_normal(p1, p2, centroid)
    move_distance = edge_length * math.sqrt(3) / 2
    hex_center = (midpoint[0] + normal[0] * move_distance,
                  midpoint[1] + normal[1] * move_distance)

    # Generate vertices
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])  # Edge angle
    hex_vertices = []
    for i in range(1, 7):
        angle_offset = angle + i * (math.pi / 3)  # 60 degree increments
        vertex = (hex_center[0] + edge_length * math.cos(angle_offset),
                  hex_center[1] + edge_length * math.sin(angle_offset))
        hex_vertices.append(vertex)

    return hex_center, hex_vertices


def is_point_in_hexagons(point, all_centers):
    """Check if point is near any existing hexagon center."""
    return any(math.dist(point, other) <= 22 for other in all_centers)


def is_majority_white(image, center, radius, ratio):
    """Check if area around center is mostly white pixels."""
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, int(radius), 255, -1)
    circle_region = cv2.bitwise_and(image, image, mask=mask)
    circle_pixels = circle_region[mask == 255]
    white_pixels = np.sum(circle_pixels == 255)
    total_pixels = len(circle_pixels)
    if total_pixels == 0:
        return False
    return white_pixels / total_pixels > ratio


def adjust_edge_length(p1, p2, target_length):
    """Adjust edge length to target value."""
    current_length = calculate_distance(p1, p2)
    scale_factor = target_length / current_length
    midpoint = calculate_midpoint(p1, p2)
    p1_new = (midpoint[0] + (p1[0] - midpoint[0]) * scale_factor,
              midpoint[1] + (p1[1] - midpoint[1]) * scale_factor)
    p2_new = (midpoint[0] + (p2[0] - midpoint[0]) * scale_factor,
              midpoint[1] + (p2[1] - midpoint[1]) * scale_factor)
    return p1_new, p2_new


def traverse_hexagon_edges(vertices, image, all_vertices, all_centers,
                           index, radius, ratio, mean_edge_len, debug):
    """Traverse hexagon edges to find adjacent hexagons."""
    centroid = calculate_centroid(vertices)
    sorted_vertices = sorted(vertices, key=lambda p: calculate_angle(centroid, p))
    counter = 0

    for i in range(len(sorted_vertices)):
        p1 = sorted_vertices[i]
        p2 = sorted_vertices[(i + 1) % len(sorted_vertices)]
        new_center = generate_new_center(p1, p2, mean_edge_len, centroid)
        new_center = [int(item) for item in new_center]

        if (not is_point_in_hexagons(new_center, all_centers) and
                is_majority_white(image, new_center, radius, ratio)):

            p1, p2 = adjust_edge_length(p1, p2, mean_edge_len)
            sorted_vertices[i] = p1
            sorted_vertices[(i + 1) % len(sorted_vertices)] = p2
            new_center, new_hexagon = generate_hexagon_around_edge(p1, p2, mean_edge_len, centroid)
            new_center = [int(item) for item in new_center]
            new_hexagon = [[int(x), int(y)] for x, y in new_hexagon]
            all_centers.append(new_center)
            all_vertices.append(new_hexagon)

            if debug:
                all_vertices_2 = [item for sublist in all_vertices for item in sublist]
                all_vertices_2 = merge_close_points(all_vertices_2, 8)
                cv2.imwrite(f'./steps/{index}_{counter}.png',
                            process_locs(transform_coor(all_vertices_2, [640, 640]), [640, 640]))
            counter += 1


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cluster_points(points, threshold):
    """Cluster points based on distance threshold."""
    clusters = []
    for point in points:
        added = False
        for cluster in clusters:
            for p in cluster:
                if euclidean_distance(p, point) <= threshold:
                    cluster.append(point)
                    added = True
                    break
            if added:
                break
        if not added:
            clusters.append([point])
    return clusters


def calculate_centers(clusters):
    """Calculate centers of point clusters."""
    centers = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        x_sum = sum(p[0] for p in cluster)
        y_sum = sum(p[1] for p in cluster)
        center = (x_sum / len(cluster), y_sum / len(cluster))
        centers.append(center)
    return centers


def merge_close_points(points, threshold):
    """Merge close points by clustering and averaging."""
    points = [tuple(point) for point in points]
    clusters = cluster_points(points, threshold)
    centers = calculate_centers(clusters)
    centers = [list(center) for center in centers]
    return centers


def gaussian_kernel(size):
    """Generate 2D Gaussian kernel."""
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel_1d = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel = kernel_1d * kernel_1d.T
    kernel = (kernel - np.min(kernel)) / np.max(kernel) * 255
    return kernel


def process_locs(locs, resize_size, kernel_size=15):
    """Process locations into an image with Gaussian blobs."""
    target_image = np.zeros(resize_size)
    for loc in locs:
        x, y = loc[0] + resize_size[1] / 10, loc[1] + resize_size[0] / 10
        x, y = int(5 * x), int(5 * y)
        b_x = max(0, x - kernel_size // 2)
        b_y = max(0, resize_size[0] - 1 - (y + kernel_size // 2))
        e_x = min(resize_size[1] - 1, x + kernel_size // 2 + 1)
        e_y = min(resize_size[0] - 1,
                  resize_size[0] - 1 - (y - kernel_size // 2 - 1))
        kernel = gaussian_kernel(kernel_size)
        target_image[b_y:e_y, b_x:e_x] = np.maximum(kernel, target_image[b_y:e_y, b_x:e_x])
    return target_image


def transform_coor(locs, image_size):
    """Transform coordinates relative to image center."""
    new_locs = []
    for loc in locs:
        new_locs.append([(loc[0] - image_size[1] / 2) / 5,
                         -(loc[1] - image_size[0] / 2) / 5])
    return new_locs


def find_nearby_points(point, candidates, threshold=1.0):
    """Find candidate points within threshold distance."""
    close_points = []
    for i, candidate in enumerate(candidates):
        if np.linalg.norm(np.array(point) - np.array(candidate)) < threshold:
            close_points.append(i)
    return close_points


def unify_shared_edges(hexagons, threshold=4):
    """Unify coordinates of shared edges between hexagons."""
    all_points = {}  # Track points and their hexagon/position

    # Collect all points
    for h, hexagon in enumerate(hexagons):
        for p, point in enumerate(hexagon):
            all_points[(h, p)] = point

    # Find and merge close points
    for (h1, p1), point1 in all_points.items():
        for (h2, p2), point2 in all_points.items():
            if h1 != h2 and (h1, p1) < (h2, p2):  # Avoid same hexagon or duplicates
                if np.linalg.norm(np.array(point1) - np.array(point2)) < threshold:
                    # Average and update
                    avg_point = ((np.array(point1) + np.array(point2)) / 2).tolist()
                    hexagons[h1][p1] = avg_point
                    hexagons[h2][p2] = avg_point

    return hexagons


def is_parallel(v1, v2, angle_threshold=10):
    """Check if two vectors are parallel within threshold."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)
    return angle < angle_threshold or (180 - angle) < angle_threshold


def distance_between_points(p1, p2):
    """Calculate distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def midpoint(point1, point2):
    """Calculate midpoint between two points."""
    return (np.array(point1) + np.array(point2)) / 2


def distance_between_lines(p1, p2, q1, q2):
    """Calculate distance between line midpoints."""
    midpoint_p = midpoint(p1, p2)
    midpoint_q = midpoint(q1, q2)
    return np.linalg.norm(midpoint_p - midpoint_q)


def merge_edges(p1, p2, q1, q2):
    """Merge two edges by averaging corresponding points."""
    p1, p2, q1, q2 = np.array(p1), np.array(p2), np.array(q1), np.array(q2)
    # Check point correspondence
    if np.linalg.norm(p1 - q1) + np.linalg.norm(p2 - q2) <= np.linalg.norm(p1 - q2) + np.linalg.norm(p2 - q1):
        new_start, new_end = (p1 + q1) / 2, (p2 + q2) / 2
        return (new_start.tolist(), new_end.tolist()), True  # p1 matches q1
    else:
        new_start, new_end = (p1 + q2) / 2, (p2 + q1) / 2
        return (new_start.tolist(), new_end.tolist()), False  # p1 matches q2


def find_and_merge_parallel_edges(hexagons, distance_threshold=1.0, angle_threshold=10):
    """Find and merge parallel edges in hexagons."""
    merged_hexagons = []

    for i, hexagon in enumerate(hexagons):
        new_hexagon = list(hexagon)  # Copy current hexagon
        for j in range(len(hexagon)):
            p1, p2 = hexagon[j], hexagon[(j + 1) % len(hexagon)]
            edge_dir = np.array(p2) - np.array(p1)

            for k, other_hexagon in enumerate(hexagons):
                if i == k:
                    continue  # Skip same hexagon

                for l in range(len(other_hexagon)):
                    q1, q2 = other_hexagon[l], other_hexagon[(l + 1) % len(other_hexagon)]
                    other_edge_dir = np.array(q2) - np.array(q1)

                    if (is_parallel(edge_dir, other_edge_dir, angle_threshold) and
                            distance_between_lines(p1, p2, q1, q2) < distance_threshold):

                        # Merge edges
                        new_edge, corresponds = merge_edges(p1, p2, q1, q2)

                        if corresponds:
                            # p1 matches q1
                            new_hexagon[j] = new_edge[0]
                            new_hexagon[(j + 1) % len(hexagon)] = new_edge[1]
                            hexagons[k][l] = new_edge[0]
                            hexagons[k][(l + 1) % len(other_hexagon)] = new_edge[1]
                        else:
                            # p1 matches q2
                            new_hexagon[j] = new_edge[0]
                            new_hexagon[(j + 1) % len(hexagon)] = new_edge[1]
                            hexagons[k][l] = new_edge[1]
                            hexagons[k][(l + 1) % len(other_hexagon)] = new_edge[0]

        merged_hexagons.append(new_hexagon)

    return merged_hexagons


def find_edges(hexagon, min_length=12, max_length=17):
    """Find edges in hexagon within length range."""
    edges = []
    num_points = len(hexagon)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(np.array(hexagon[i]) - np.array(hexagon[j]))
            if min_length <= distance <= max_length:
                edges.append((i, j))
    return edges


def build_ordered_hexagon(hexagon, edges):
    """Build ordered hexagon from edges."""
    if not edges or len(edges) < 6:
        return []  # Not enough edges

    ordered_points = []
    used_edges = set()
    current_point = edges[0][0]  # Start with first edge
    ordered_points.append(current_point)

    # Build hexagon by following edges
    while len(ordered_points) < 6:
        found = False
        for edge in edges:
            if edge in used_edges:
                continue
            if current_point in edge:
                next_point = edge[1] if edge[0] == current_point else edge[0]
                if next_point not in ordered_points:
                    ordered_points.append(next_point)
                    used_edges.add(edge)
                    current_point = next_point
                    found = True
                    break
        if not found:
            break  # Can't complete hexagon

    return [hexagon[i] for i in ordered_points] if len(ordered_points) == 6 else []


def reorder_hexagons(hexagons, min_length=12, max_length=17):
    """Reorder hexagon vertices based on edge connections."""
    ordered_hexagons = []
    for hexagon in hexagons:
        edges = find_edges(hexagon, min_length, max_length)
        ordered_hexagon = build_ordered_hexagon(hexagon, edges)
        if ordered_hexagon:
            ordered_hexagons.append(ordered_hexagon)
        else:
            ordered_hexagons.append(hexagon)  # Keep original if can't reorder
    return ordered_hexagons
