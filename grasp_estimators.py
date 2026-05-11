import numpy as np
import cv2
from stretch4_gripper_modeling_and_control import gripper_camera as gc
import visual_servo_fsm_params as vp

class GraspTarget:
    def __init__(self, grasp_width, grasp_axis_3d, center_3d, left_contact_3d, right_contact_3d):
        self.grasp_width = grasp_width
        self.grasp_axis_3d = grasp_axis_3d # Normalized vector pointing from center to one of the contacts
        self.center_3d = center_3d
        self.left_contact_3d = left_contact_3d
        self.right_contact_3d = right_contact_3d

class GraspEstimator:
    def __init__(self):
        pass
        
    def estimate(self, mask, depth_image, rgb_camera_info, object_center_3d) -> GraspTarget:
        raise NotImplementedError("Base class must implement estimate()")

class EllipsoidGraspEstimator(GraspEstimator):
    def __init__(self, use_3d=False):
        super().__init__()
        self.use_3d = use_3d

    def estimate(self, mask, depth_image, rgb_camera_info, object_center_3d) -> GraspTarget:
        if mask is None or object_center_3d is None:
            return None
            
        mask_2d = mask.squeeze()
        mask_8u = (mask_2d > 0).astype(np.uint8)
        
        if self.use_3d:
            if depth_image is None:
                return None
            
            # Extract 3D points
            ys, xs = np.where(mask_8u > 0)
            if len(xs) < 10:
                return None
                
            depths = depth_image[ys, xs] / 1000.0
            valid = depths > 0.0
            xs = xs[valid]
            ys = ys[valid]
            depths = depths[valid]
            
            if len(xs) < 10:
                return None
            
            # Convert to 3D point cloud
            pts_3d = []
            for i in range(len(xs)):
                pts_3d.append(gc.pixel_to_3d((xs[i], ys[i]), depths[i], rgb_camera_info))
            pts_3d = np.array(pts_3d)
            
            # PCA on 3D points to find the ellipsoid axes
            mean_pt = np.mean(pts_3d, axis=0)
            centered_pts = pts_3d - mean_pt
            cov = np.cov(centered_pts, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Grasp along the minor axis
            # For 2.5D point clouds from a camera, the 3rd component is usually depth noise
            # We use the 2nd principal component (minor axis in the image plane roughly)
            minor_axis_3d = eigenvectors[:, 1]
            if minor_axis_3d[0] < 0: # Ensure consistent X convention
                minor_axis_3d = -minor_axis_3d
                
            # If the length of the minor axis is not significantly smaller than the length of the major axis,
            # the target grasp orientation should be horizontal, corresponding with a wrist roll angle of 0.0.
            if eigenvalues[0] > 0 and np.sqrt(max(0, eigenvalues[1])) / np.sqrt(max(0, eigenvalues[0])) > vp.CIRCULAR_MASK_AXIS_RATIO_THRESHOLD:
                minor_axis_3d = np.array([1.0, 0.0, 0.0])
                
            # Project points onto minor axis to find extent
            projections = np.dot(centered_pts, minor_axis_3d)
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            
            left_contact = object_center_3d + minor_axis_3d * min_proj
            right_contact = object_center_3d + minor_axis_3d * max_proj
            
            grasp_width = max_proj - min_proj
            
            return GraspTarget(grasp_width, minor_axis_3d, object_center_3d, left_contact, right_contact)

        else:
            # 2D Mask fitting
            contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) < 5:
                return None
                
            ellipse = cv2.fitEllipseAMS(largest_contour)
            (cx, cy), (MA, ma), angle = ellipse # MA corresponds to width/minor. ma is height/major.
            
            # If the length of the minor axis is not significantly smaller than the length of the major axis,
            # the target grasp orientation should be horizontal, corresponding with a wrist roll angle of 0.0.
            if ma > 0 and (MA / ma) > vp.CIRCULAR_MASK_AXIS_RATIO_THRESHOLD:
                angle = 0.0
            
            # Convert angle to 2D vector
            # The angle is returned in degrees, measured clockwise from the vertical axis (Y-axis points down).
            # If the ellipse is horizontal, angle is 90.
            angle_rad = np.deg2rad(angle)
            
            # The minor axis is along (cos(angle), sin(angle))
            # Wait, OpenCV fitEllipse angle is angle of the MAJOR axis from the vertical (Y-axis).
            # The minor axis is perpendicular to the major axis.
            minor_angle_rad = angle_rad + np.pi / 2.0
            
            dir_x = np.sin(minor_angle_rad)
            dir_y = -np.cos(minor_angle_rad)
            
            # Half minor axis
            half_minor = MA / 2.0
            
            # Center the grasp on the 2D segmentation mask centroid
            cx, cy = gc.pixel_from_3d(object_center_3d, rgb_camera_info)
            
            pt1_2d = (cx + dir_x * half_minor, cy + dir_y * half_minor)
            pt2_2d = (cx - dir_x * half_minor, cy - dir_y * half_minor)
            
            depth = object_center_3d[2]
            
            pt1_3d = gc.pixel_to_3d(pt1_2d, depth, rgb_camera_info)
            pt2_3d = gc.pixel_to_3d(pt2_2d, depth, rgb_camera_info)
            
            axis_3d = pt1_3d - pt2_3d
            width = np.linalg.norm(axis_3d)
            if width < 1e-4:
                return None
                
            axis_3d = axis_3d / width
            
            return GraspTarget(width, axis_3d, object_center_3d, pt2_3d, pt1_3d)

class DepthProfileGraspEstimator(GraspEstimator):
    def __init__(self):
        super().__init__()
        # Use an underlying 2D ellipsoid to find the principal minor axis, then trace depths
        self.base_estimator = EllipsoidGraspEstimator(use_3d=False)
        
    def estimate(self, mask, depth_image, rgb_camera_info, object_center_3d) -> GraspTarget:
        base_target = self.base_estimator.estimate(mask, depth_image, rgb_camera_info, object_center_3d)
        if base_target is None or depth_image is None:
            return None
            
        mask_2d = mask.squeeze()
        h, w = mask_2d.shape
        
        # We trace along the 2D minor axis to find the true edge
        cx, cy = gc.pixel_from_3d(object_center_3d, rgb_camera_info)
        
        axis_pt1_2d = gc.pixel_from_3d(base_target.left_contact_3d, rgb_camera_info)
        axis_pt2_2d = gc.pixel_from_3d(base_target.right_contact_3d, rgb_camera_info)
        
        dir_vector = axis_pt2_2d - axis_pt1_2d
        norm = np.linalg.norm(dir_vector)
        if norm < 1e-4:
            return base_target
        dir_2d = dir_vector / norm
        
        # Raycast outwards from center until we hit the mask edge, and take that depth
        def find_edge_3d(start_x, start_y, dx, dy, max_steps=400):
            x, y = start_x, start_y
            last_valid_z = object_center_3d[2]
            last_x, last_y = start_x, start_y
            
            for _ in range(max_steps):
                x += dx
                y += dy
                ix, iy = int(np.round(x)), int(np.round(y))
                
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                    
                if mask_2d[iy, ix] == 0:
                    break
                    
                depth_val = depth_image[iy, ix] / 1000.0
                if depth_val > 0.0:
                    last_valid_z = depth_val
                last_x, last_y = ix, iy
                
            return gc.pixel_to_3d((last_x, last_y), last_valid_z, rgb_camera_info)
            
        left_3d = find_edge_3d(cx, cy, -dir_2d[0], -dir_2d[1])
        right_3d = find_edge_3d(cx, cy, dir_2d[0], dir_2d[1])
        
        grasp_axis = right_3d - left_3d
        width = np.linalg.norm(grasp_axis)
        if width > 1e-4:
            grasp_axis = grasp_axis / width
        else:
            grasp_axis = base_target.grasp_axis_3d
            
        return GraspTarget(width, grasp_axis, object_center_3d, left_3d, right_3d)

