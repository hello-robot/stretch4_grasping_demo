import cv2
import visualization_utils as vu

class UIManager:
    def __init__(self, args):
        self.args = args
        self.windows_resized = set()
        self.molmo_window_height = 0

    def create_molmo_ui(self, scaled_color_image, generated_text, object_description, pt_scaled, state_name):
        return vu.create_text_visualization(scaled_color_image, generated_text, object_description, point=pt_scaled, state_name=state_name)
        
    def overlay_tracking(self, scaled_color_image, last_mask, centroid, state_name, median_depth, v_desired, rot_change, target_grasp, scaled_camera_info, estimator_name):
        if self.args.display_scale != 1.0:
            import numpy as np
            scaled_mask = cv2.resize(last_mask.astype(np.uint8), (0, 0), fx=self.args.display_scale, fy=self.args.display_scale, interpolation=cv2.INTER_NEAREST)
            scaled_centroid = (centroid[0] * self.args.display_scale, centroid[1] * self.args.display_scale)
        else:
            scaled_mask = last_mask
            scaled_centroid = centroid
            
        display_image = vu.overlay_mask(scaled_color_image, scaled_mask, centroid=scaled_centroid)
        
        depth_str = f"{median_depth:.2f}m" if median_depth is not None else "N/A"
        cv2.putText(display_image, f"State: {state_name} | Depth: {depth_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if v_desired is not None and rot_change is not None:
            cv2.putText(display_image, f"v=[{v_desired[0]:.2f}, {v_desired[1]:.2f}, {v_desired[2]:.2f}] | rot=[{rot_change[0]:.2f}, {rot_change[1]:.2f}, {rot_change[2]:.2f}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if target_grasp is not None:
            display_image = vu.draw_target_grasp(display_image, target_grasp, scaled_camera_info)
            width_str = f"Width: {target_grasp.grasp_width*100:.1f}cm"
            cv2.putText(display_image, f"Grasp: {estimator_name} | {width_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        return display_image

    def draw_fingertip_markers(self, display_image, fingertip_center_2d, estimation_method):
        if fingertip_center_2d is not None:
            scaled_finger_center = (int(fingertip_center_2d[0] * self.args.display_scale), int(fingertip_center_2d[1] * self.args.display_scale))
            cv2.circle(display_image, scaled_finger_center, 8, (255, 165, 0), -1)
            cv2.circle(display_image, scaled_finger_center, 8, (0, 0, 0), 2)
            if estimation_method is not None:
                cv2.putText(display_image, estimation_method.name, (scaled_finger_center[0] + 10, scaled_finger_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        return display_image

    def show_window(self, win_name, display_image):
        if display_image is None:
            return
            
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        is_first_time = win_name not in self.windows_resized
        
        cv2.imshow(win_name, display_image)
        
        title_bar_height = 40
        if win_name == "Molmo Object Initialization":
            cv2.moveWindow(win_name, 0, 0)
            self.molmo_window_height = display_image.shape[0] + title_bar_height
        elif win_name in ["Visual Servoing Output", "Camera Stream"]:
            y_pos = self.molmo_window_height if self.molmo_window_height > 0 else 0
            cv2.moveWindow(win_name, 0, y_pos)
            
        if is_first_time:
            cv2.resizeWindow(win_name, display_image.shape[1], display_image.shape[0])
            self.windows_resized.add(win_name)
