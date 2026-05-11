import cv2
import numpy as np

def compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, kp_yaw, kp_pitch):
    if fingertip_center_3d is None or object_center_3d is None or plane_normal is None or curr_wrist_roll is None:
        return [0.0, 0.0, 0.0]

    n = np.array(plane_normal)
    # Ensure normal points "down" towards positive Y
    if n[1] < 0:
        n = -n

    V = np.array(object_center_3d) - np.array(fingertip_center_3d)
    V_norm = np.linalg.norm(V)
    if V_norm < 1e-6:
        return [0.0, 0.0, 0.0]

    # Pitch error: Angle between V and the plane
    err_y = np.arcsin(np.clip(np.dot(V, n) / V_norm, -1.0, 1.0))
    
    # Gripper vector G: The camera Z-axis projected onto the plane
    cam_z = np.array([0.0, 0.0, 1.0])
    G_raw = cam_z - np.dot(cam_z, n) * n
    G_norm = np.linalg.norm(G_raw)
    if G_norm < 1e-6:
        return [0.0, 0.0, 0.0]
    G = G_raw / G_norm
    
    # Yaw error: Angle between G and V projected onto the plane
    V_proj = V - np.dot(V, n) * n
    if np.linalg.norm(V_proj) < 1e-6:
        err_x = 0.0
    else:
        cross_prod = np.cross(G, V_proj)
        err_x = np.arctan2(np.dot(cross_prod, n), np.dot(G, V_proj))
        
    # 2D Rotation by the camera's roll angle to map the error vectors back to the joint frame
    yaw_error = err_x * np.cos(curr_wrist_roll) - err_y * np.sin(curr_wrist_roll)
    pitch_error = err_x * np.sin(curr_wrist_roll) + err_y * np.cos(curr_wrist_roll)
        
    # Apply a deadband to the angular errors to prevent unending limit cycle oscillations
    # caused by stick-slip friction, latency, and backlash in the physical joints.
    # 0.035 rad is ~2.0 degrees (approx 1cm error at 30cm distance).
    DEADBAND_RAD = 0.035
    if abs(yaw_error) < DEADBAND_RAD:
        yaw_error = 0.0
    if abs(pitch_error) < DEADBAND_RAD:
        pitch_error = 0.0
        
    yaw_cmd = -kp_yaw * yaw_error
    pitch_cmd = -kp_pitch * pitch_error
    
    yaw_cmd = max(min(yaw_cmd, 1.0), -1.0)
    pitch_cmd = max(min(pitch_cmd, 1.0), -1.0)
    
    return [yaw_cmd, pitch_cmd, 0.0]

def compute_3d_align_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, target_grasp_axis, fingertip_axis, curr_wrist_roll, kp_yaw, kp_pitch, kp_roll, roll_limits=(-4.27605, 1.13446), debug=False): # [-245 deg, 65 deg]
    # Keep it perfectly centered using existing pitch/yaw math
    base_vels = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, kp_yaw, kp_pitch)
    
    yaw_cmd = base_vels[0]
    pitch_cmd = base_vels[1]
    
    if target_grasp_axis is None or fingertip_axis is None or curr_wrist_roll is None:
        return [yaw_cmd, pitch_cmd, 0.0], float('inf')
        
    # Project 3D vectors onto the camera XY plane
    gx, gy = target_grasp_axis[0], target_grasp_axis[1]
    g_norm = np.hypot(gx, gy)
    
    fx, fy = fingertip_axis[0], fingertip_axis[1]
    f_norm = np.hypot(fx, fy)
    
    if g_norm < 1e-4 or f_norm < 1e-4:
        return [yaw_cmd, pitch_cmd, 0.0], float('inf')
        
    beta = np.arctan2(gy, gx)
    alpha = np.arctan2(fy, fx)
    
    d_theta = beta - alpha
    # Wrap to [-pi, pi]
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    
    # Due to symmetry, the target can be d_theta or d_theta + pi
    # e1 = path 1, e2 = path 2
    e1 = d_theta
    e2 = d_theta + np.pi
    e2 = (e2 + np.pi) % (2 * np.pi) - np.pi
    
    target1 = curr_wrist_roll + e1
    target2 = curr_wrist_roll + e2
    
    valid1 = roll_limits[0] <= target1 <= roll_limits[1]
    valid2 = roll_limits[0] <= target2 <= roll_limits[1]
    
    # Decide which error path gives the best valid destination!
    chosen_error = None
    if valid1 and valid2:
        chosen_error = e1 if abs(e1) < abs(e2) else e2
    elif valid1:
        chosen_error = e1
    elif valid2:
        chosen_error = e2
    else:
        # If neither is globally attainable due to stiff limits, clip to bounds!
        # Pick the valid boundary that's closest to the target.
        dist1 = min(abs(target1 - roll_limits[0]), abs(target1 - roll_limits[1]))
        dist2 = min(abs(target2 - roll_limits[0]), abs(target2 - roll_limits[1]))
        
        if dist1 < dist2:
            bound = roll_limits[1] if target1 > roll_limits[1] else roll_limits[0]
            chosen_error = bound - curr_wrist_roll
        else:
            bound = roll_limits[1] if target2 > roll_limits[1] else roll_limits[0]
            chosen_error = bound - curr_wrist_roll
            
    # Scale positive angular error into velocity (a positive error in OpenCV's XY plane is a CW rotation, which matches the positive joint movement mapping CW)
    roll_cmd = kp_roll * chosen_error
    if debug:
        print(f"ALIGNMENT DEBUG -- gx,gy: {gx:.2f},{gy:.2f} | fx,fy: {fx:.2f},{fy:.2f} | beta: {beta:.2f} | alpha: {alpha:.2f} | d_theta: {d_theta:.2f} | curr_roll: {curr_wrist_roll:.2f} | target_joint: {target1:.2f} | roll_cmd: {roll_cmd:.2f}")
    roll_cmd = max(min(roll_cmd, 1.0), -1.0)
    
    return [yaw_cmd, pitch_cmd, roll_cmd], chosen_error


def get_median_object_depth(last_mask, depth_image):
    if depth_image is None or last_mask is None:
        return None
    depth_meters = depth_image / 1000.0
    object_depths = depth_meters[(last_mask > 0)]
    valid_depths = object_depths[object_depths > 0.0]
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    return None

def compute_look_at_rot_velocities(last_mask, W, H, kp_yaw, kp_pitch, fingertip_center_2d=None, image_servo_mode='image_center_to_object'):
    mask_2d = last_mask.squeeze()
    mask_8u = (mask_2d > 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area to ignore any tracking noise
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx_mask = M["m10"] / M["m00"]
            cy_mask = M["m01"] / M["m00"]
        else:
            cx_mask = W / 2.0
            cy_mask = H / 2.0
    else:
        cx_mask = W / 2.0
        cy_mask = H / 2.0

    if image_servo_mode == 'fingertip_to_object' and fingertip_center_2d is not None:
        target_x = fingertip_center_2d[0]
        target_y = fingertip_center_2d[1]
    else:
        # 'image_center_to_object' or fallback
        target_x = W / 2.0
        target_y = H / 2.0

    # In typical image coords: +X is right, +Y is down.
    error_x = (cx_mask - target_x) / W
    error_y = (cy_mask - target_y) / H
    
    # Positive error_x (object to right) requires camera to rotate right.
    # In standard gripper kinematics (yaw around Z up), positive yaw is counter-clockwise (rotating left).
    # Thus, we send negative yaw to rotate right towards the object.
    yaw_cmd = -kp_yaw * error_x
    
    # Positive error_y (object down) requires camera to rotate down.
    # In standard gripper kinematics (pitch around Y right), positive pitch rotates up.
    # Thus, we send negative pitch to rotate down towards the object.
    pitch_cmd = -kp_pitch * error_y
    
    # Bound velocities to [-1.0, 1.0]
    yaw_cmd = max(min(yaw_cmd, 1.0), -1.0)
    pitch_cmd = max(min(pitch_cmd, 1.0), -1.0)
    
    rot_change = [yaw_cmd, pitch_cmd, 0.0]
    return rot_change, (cx_mask, cy_mask)
