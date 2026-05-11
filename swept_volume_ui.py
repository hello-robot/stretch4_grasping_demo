import cv2
import numpy as np
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af
from stretch4_gripper_modeling_and_control import gripper_camera as gc
from stretch4_gripper_modeling_and_control.swept_volume_model import SweptVolumeModel

class SweptVolumeProcessor:
    def __init__(self, visualizer, args):
        self.visualizer = visualizer
        self.args = args

    def segment_depth_image(self, color_image, depth_image, rgb_camera_info, vis_fingertips):
        h, w = color_image.shape[:2]
        min_depth_img = np.full((h, w), float('inf'), dtype=np.float32)
        max_depth_img = np.full((h, w), float('-inf'), dtype=np.float32)
        tube_mask_2d = np.zeros((h, w), dtype=np.uint8)
        has_valid_depth_range = False
        
        for f_side in ['left', 'right']:
            if f_side in vis_fingertips:
                sv_model = SweptVolumeModel(self.visualizer, f_side, vis_fingertips[f_side], self.args.swept_volume_to_pos)
                if sv_model.valid:
                    pcts = sv_model.get_sampled_pcts(sampling_method=self.args.swept_volume_sampling_method, num_samples=self.args.swept_volume_samples)
                    prev_pts_2d = None
                    prev_f_min = None
                    prev_f_max = None
                    for pct in pcts:
                        pos, rot = sv_model.get_frame(pct)
                        if pos is not None:
                            x_axis = rot[:, 0]
                            y_axis = rot[:, 1]
                            dz = af.suctioncup_radius * np.sqrt(x_axis[2]**2 + y_axis[2]**2)
                            f_min = pos[2] - dz
                            f_max = pos[2] + dz
                            
                        pts_3d = sv_model.get_circle_points(pct)
                        if pts_3d is not None:
                            pts_2d = [np.round(gc.pixel_from_3d(p, rgb_camera_info)).astype(np.int32) for p in pts_3d]
                            
                            if prev_pts_2d is not None:
                                all_pts = np.vstack((prev_pts_2d, pts_2d))
                                hull = cv2.convexHull(all_pts)
                                cv2.fillPoly(tube_mask_2d, [hull], 255, lineType=cv2.LINE_AA)
                                
                                seg_mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(seg_mask, [hull], 255, lineType=cv2.LINE_AA)
                                b_mask = seg_mask > 0
                                
                                seg_min = min(prev_f_min, f_min)
                                seg_max = max(prev_f_max, f_max)
                                
                                min_depth_img[b_mask] = np.minimum(min_depth_img[b_mask], seg_min)
                                max_depth_img[b_mask] = np.maximum(max_depth_img[b_mask], seg_max)
                                has_valid_depth_range = True
                                
                            prev_pts_2d = pts_2d
                            prev_f_min = f_min
                            prev_f_max = f_max
        
        segmented_rgb = color_image.copy()
        if depth_image is not None and has_valid_depth_range:
            depth_meters = depth_image / 1000.0
            mask = (depth_meters >= min_depth_img) & (depth_meters <= max_depth_img) & (depth_meters > 0.0)
            mask = mask & (tube_mask_2d > 0)
            
            dimmed_bg = cv2.addWeighted(color_image, self.args.bg_visibility, np.zeros_like(color_image), 0, 0)
            if self.args.fg_amplification > 0.0:
                alpha = 1.0 + self.args.fg_amplification
                beta = int(127 * self.args.fg_amplification)
                bright_fg = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)
            else:
                bright_fg = color_image
            
            segmented_rgb = np.where(mask[:, :, np.newaxis], bright_fg, dimmed_bg)

        return segmented_rgb

    def draw_swept_volume_overlay(self, display_image, vis_fingertips, scaled_camera_info):
        overlay = display_image.copy()
        draw_occurred = False
        for f_side in ['left', 'right']:
            if f_side in vis_fingertips:
                sv_model = SweptVolumeModel(self.visualizer, f_side, vis_fingertips[f_side], self.args.swept_volume_to_pos)
                if sv_model.valid:
                    pcts = sv_model.get_sampled_pcts(sampling_method=self.args.swept_volume_sampling_method, num_samples=self.args.swept_volume_samples)
                    prev_pts_2d = None
                    for pct in pcts:
                        pos, rot = sv_model.get_frame(pct)
                        pts_3d = sv_model.get_circle_points(pct)
                        if pts_3d is not None and pos is not None:
                            pts_2d = [np.round(gc.pixel_from_3d(p, scaled_camera_info)).astype(np.int32) for p in pts_3d]
                            
                            if prev_pts_2d is not None:
                                all_pts = np.vstack((prev_pts_2d, pts_2d))
                                hull = cv2.convexHull(all_pts)
                                cv2.fillPoly(overlay, [hull], (180, 140, 70), lineType=cv2.LINE_AA)
                                
                            prev_pts_2d = pts_2d
                            draw_occurred = True
        
        if draw_occurred:
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)
            
            if self.args.swept_volume_mesh_visibility > 0.0:
                wire_overlay = display_image.copy()
                outline_color = (180, 120, 40)
                for f_side in ['left', 'right']:
                    if f_side in vis_fingertips:
                        sv_model = SweptVolumeModel(self.visualizer, f_side, vis_fingertips[f_side], self.args.swept_volume_to_pos)
                        if sv_model.valid:
                            pcts = sv_model.get_sampled_pcts(sampling_method=self.args.swept_volume_sampling_method, num_samples=self.args.swept_volume_samples)
                            prev_pts_2d = None
                            prev_pts_3d = None
                            prev_pos = None
                            for pct in pcts:
                                pos, rot = sv_model.get_frame(pct)
                                pts_3d = sv_model.get_circle_points(pct)
                                if pts_3d is not None and pos is not None:
                                    pts_2d = [np.round(gc.pixel_from_3d(p, scaled_camera_info)).astype(np.int32) for p in pts_3d]
                                    num_points = len(pts_3d)
                                    
                                    for i in range(num_points):
                                        next_i = (i + 1) % num_points
                                        mid_pt_3d = (pts_3d[i] + pts_3d[next_i]) / 2.0
                                        normal_3d = mid_pt_3d - pos
                                        if np.dot(normal_3d, mid_pt_3d) < 0:
                                            cv2.line(wire_overlay, pts_2d[i], pts_2d[next_i], outline_color, 2, lineType=cv2.LINE_AA)
                                            
                                    if prev_pts_2d is not None:
                                        for i in range(num_points):
                                            mid_pt_3d = (pts_3d[i] + prev_pts_3d[i]) / 2.0
                                            mid_pos = (pos + prev_pos) / 2.0
                                            normal_3d = mid_pt_3d - mid_pos
                                            if np.dot(normal_3d, mid_pt_3d) < 0:
                                                cv2.line(wire_overlay, prev_pts_2d[i], pts_2d[i], outline_color, 1, lineType=cv2.LINE_AA)
                                                
                                    prev_pts_2d = pts_2d
                                    prev_pts_3d = pts_3d
                                    prev_pos = pos
                
                cv2.addWeighted(wire_overlay, self.args.swept_volume_mesh_visibility, display_image, 1 - self.args.swept_volume_mesh_visibility, 0, display_image)
        return display_image
