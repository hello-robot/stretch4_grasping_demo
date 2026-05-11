import argparse
import cv2
import copy
import numpy as np
import textwrap

def add_display_scale_argument(parser):
    """Adds the --display_scale argument to an existing argparse.ArgumentParser."""
    parser.add_argument('--display_scale', type=float, default=1.0, 
                        help='Scale factor for the OpenCV visualization image (1.0 = native, 2.0 = double size, 0.5 = half size).')

def apply_display_scale(image, scale, camera_info=None):
    """
    Scales the image by the given scale factor prior to annotation.
    If camera_info is provided, scales the intrinsic camera matrix logically to match the new image dimensions.
    Returns:
        scaled_image: The resized image.
        scaled_camera_info (optional): The mathematically matched visual camera info dict.
    """
    if scale == 1.0:
        if camera_info is not None:
            return image, camera_info
        return image

    scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    if camera_info is not None:
        scaled_camera_info = copy.deepcopy(camera_info)
        # Scale focal lengths and principal center points linearly
        scaled_camera_info['camera_matrix'][0, 0] *= scale
        scaled_camera_info['camera_matrix'][1, 1] *= scale
        scaled_camera_info['camera_matrix'][0, 2] *= scale
        scaled_camera_info['camera_matrix'][1, 2] *= scale
        return scaled_image, scaled_camera_info
        
    return scaled_image

def overlay_mask(image, mask, centroid=None):
    overlay = image.copy()
    color_mask = [255, 0, 0] # Blue mask
    
    mask_bool = mask.astype(bool).squeeze()
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color_mask) * 0.5
    
    if centroid is not None:
        cv2.circle(overlay, (int(centroid[0]), int(centroid[1])), radius=8, color=(0, 255, 0), thickness=-1)
        
    return overlay

def create_text_visualization(image, text, object_description, point=None, state_name=None):
    H, W = image.shape[:2]
    text_panel_h = 130
    panel = np.zeros((text_panel_h, W, 3), dtype=np.uint8)
    
    text1 = f"State: {state_name if state_name else ''} | Object: "
    cv2.putText(panel, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    size_obj, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(panel, object_description, (10 + size_obj[0], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    text2 = "Molmo 2: "
    cv2.putText(panel, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    size_molmo, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    chars_per_line = max(40, int(W / 12))
    lines = textwrap.wrap(text2 + text, width=chars_per_line)
    
    y0, dy = 70, 25 
    for i, line in enumerate(lines[:3]):
        if i == 0 and line.startswith(text2):
            val = line[len(text2):]
            cv2.putText(panel, val, (10 + size_molmo[0], y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(panel, line, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    vis_img = image.copy()
    if point is not None:
        cv2.circle(vis_img, point, radius=10, color=(0, 255, 0), thickness=-1)
        cv2.circle(vis_img, point, radius=12, color=(0, 0, 255), thickness=2)
        
    return np.vstack((vis_img, panel))

def draw_target_grasp(image, grasp_target, camera_info):
    """
    Draws two parallel line segments to represent the target grasp width and orientation.
    """
    if grasp_target is None or camera_info is None:
        return image
        
    from stretch4_gripper_modeling_and_control import gripper_camera as gc
    
    # Project 3D contacts to 2D
    pt1_2d = gc.pixel_from_3d(grasp_target.left_contact_3d, camera_info)
    pt2_2d = gc.pixel_from_3d(grasp_target.right_contact_3d, camera_info)
    
    pt1 = (int(np.round(pt1_2d[0])), int(np.round(pt1_2d[1])))
    pt2 = (int(np.round(pt2_2d[0])), int(np.round(pt2_2d[1])))
    
    # Calculate 2D direction and perpendicular
    dx = pt2_2d[0] - pt1_2d[0]
    dy = pt2_2d[1] - pt1_2d[1]
    norm = np.sqrt(dx**2 + dy**2)
    
    if norm > 1e-3:
        nx = -dy / norm
        ny = dx / norm
    else:
        nx, ny = 0, 1
        
    # Scale length of the segments (e.g. 15 pixels each side)
    length = 15
    
    p1_a = (int(pt1[0] + nx * length), int(pt1[1] + ny * length))
    p1_b = (int(pt1[0] - nx * length), int(pt1[1] - ny * length))
    
    p2_a = (int(pt2[0] + nx * length), int(pt2[1] + ny * length))
    p2_b = (int(pt2[0] - nx * length), int(pt2[1] - ny * length))
    
    overlay = image.copy()
    
    # Draw contact pads as lines
    cv2.line(overlay, p1_a, p1_b, (0, 165, 255), 3, cv2.LINE_AA) # Orange lines
    cv2.line(overlay, p2_a, p2_b, (0, 165, 255), 3, cv2.LINE_AA)
    
    # Connect them with a dotted/thinner line representing the axis
    cv2.line(overlay, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
    
    return overlay
