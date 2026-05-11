import numpy as np
import vlm_utils as vlm
import sam2_utils as sam2
from stretch4_gripper_modeling_and_control import gripper_camera as gc
import visual_servo_fsm_params as vp
from perception_utils import get_median_object_depth

class PerceptionPipeline:
    def __init__(self, tracker, grasp_estimator, molmo_proc, molmo_model, args):
        self.tracker = tracker
        self.grasp_estimator = grasp_estimator
        self.molmo_proc = molmo_proc
        self.molmo_model = molmo_model
        self.args = args

    def run_molmo_detection(self, rgb_image, object_description):
        from PIL import Image
        pil_image = Image.fromarray(rgb_image)
        width, height = pil_image.size
        
        molmo_prompt = vlm.get_molmo_pointing_prompt(object_description)
        inputs = self.molmo_proc(images=[pil_image], text=molmo_prompt, return_tensors="pt")
        inputs = {k: v.to(self.molmo_model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.molmo_model.dtype)

        output_ids = self.molmo_model.generate(**inputs, max_new_tokens=200, stop_strings=["<|endoftext|>"], tokenizer=self.molmo_proc.tokenizer)
        generated_tokens = output_ids[0, inputs['input_ids'].size(1):]
        generated_text = self.molmo_proc.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"[FIND_OBJECT] Molmo response: {generated_text}")
        
        pt_scaled = None
        target_point = None
        if "not found" not in generated_text.lower():
            pts = vlm.extract_points(generated_text, width, height)
            if len(pts) > 0:
                target_point = pts[0]
                pt_scaled = (int(target_point[0] * self.args.display_scale), int(target_point[1] * self.args.display_scale))
                
        return target_point, pt_scaled, generated_text

    def update_tracking_and_grasp(self, fsm, rgb_image, depth_image, rgb_camera_info, H, W):
        last_mask, target_lost = self.tracker.update_tracking(rgb_image)
        if target_lost:
            return None, True, None

        target_grasp = None
        _, centroid = vlm.compute_look_at_rot_velocities(last_mask, W, H, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object') if getattr(vlm, 'compute_look_at_rot_velocities', None) else (None, None)
        if centroid is None:
            from perception_utils import compute_look_at_rot_velocities
            _, centroid = compute_look_at_rot_velocities(last_mask, W, H, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        if self.grasp_estimator is not None and median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)
            target_grasp = self.grasp_estimator.estimate(last_mask, depth_image, rgb_camera_info, object_center_3d)

            if target_grasp is not None:
                if fsm.current_state in [fsm.current_state.ALIGN_GRASP, fsm.current_state.PRE_GRASP]:
                    fsm.reference_y_3d = target_grasp.center_3d[1]
                    import copy
                    fsm.last_good_target_grasp = copy.deepcopy(target_grasp)
                
                elif fsm.current_state in [fsm.current_state.APPROACH_GRASP, fsm.current_state.EXECUTE_GRASP, fsm.current_state.LIFT_OBJECT, fsm.current_state.RETRACT_GRASP] and hasattr(fsm, 'reference_y_3d'):
                    fy = rgb_camera_info['camera_matrix'][1, 1]
                    cy_img = rgb_camera_info['camera_matrix'][1, 2]
                    
                    y_pixel = (fsm.reference_y_3d * fy / median_depth) + cy_img
                    y_pixel_int = int(np.round(y_pixel))
                    y_pixel_int = np.clip(y_pixel_int, 0, H - 1)
                    
                    mask_2d = last_mask.squeeze()
                    row_pixels = np.where(mask_2d[y_pixel_int, :] > 0)[0]
                    
                    if len(row_pixels) == 0:
                        if hasattr(fsm, 'last_good_target_grasp'):
                            cx, cy = gc.pixel_from_3d(fsm.last_good_target_grasp.center_3d, rgb_camera_info)
                            new_center_3d = gc.pixel_to_3d((cx, cy), median_depth, rgb_camera_info)
                            shift_3d = new_center_3d - fsm.last_good_target_grasp.center_3d
                            
                            target_grasp.center_3d = new_center_3d
                            target_grasp.left_contact_3d = fsm.last_good_target_grasp.left_contact_3d + shift_3d
                            target_grasp.right_contact_3d = fsm.last_good_target_grasp.right_contact_3d + shift_3d
                            target_grasp.grasp_width = fsm.last_good_target_grasp.grasp_width
                            target_grasp.grasp_axis_3d = fsm.last_good_target_grasp.grasp_axis_3d
                    else:
                        left_x = row_pixels[0]
                        right_x = row_pixels[-1]
                        center_x = (left_x + right_x) / 2.0
                        
                        if depth_image is not None:
                            left_depth = depth_image[y_pixel_int, left_x] / 1000.0
                            right_depth = depth_image[y_pixel_int, right_x] / 1000.0
                            center_depth = depth_image[y_pixel_int, int(center_x)] / 1000.0
                            
                            if left_depth <= 0.0: left_depth = median_depth
                            if right_depth <= 0.0: right_depth = median_depth
                            if center_depth <= 0.0: center_depth = median_depth
                        else:
                            left_depth = right_depth = center_depth = median_depth

                        new_center_3d = gc.pixel_to_3d((center_x, y_pixel_int), center_depth, rgb_camera_info)
                        left_contact_3d = gc.pixel_to_3d((left_x, y_pixel_int), left_depth, rgb_camera_info)
                        right_contact_3d = gc.pixel_to_3d((right_x, y_pixel_int), right_depth, rgb_camera_info)
                        
                        grasp_axis_3d = right_contact_3d - left_contact_3d
                        width = np.linalg.norm(grasp_axis_3d)
                        if width > 1e-4:
                            grasp_axis_3d = grasp_axis_3d / width
                        else:
                            grasp_axis_3d = target_grasp.grasp_axis_3d
                        
                        if hasattr(fsm, 'last_good_target_grasp'):
                            fsm.last_good_target_grasp.center_3d = new_center_3d
                            fsm.last_good_target_grasp.left_contact_3d = left_contact_3d
                            fsm.last_good_target_grasp.right_contact_3d = right_contact_3d
                            fsm.last_good_target_grasp.grasp_width = width
                            fsm.last_good_target_grasp.grasp_axis_3d = grasp_axis_3d
                    
                    if hasattr(fsm, 'last_good_target_grasp'):
                        target_grasp.center_3d = fsm.last_good_target_grasp.center_3d
                        target_grasp.left_contact_3d = fsm.last_good_target_grasp.left_contact_3d
                        target_grasp.right_contact_3d = fsm.last_good_target_grasp.right_contact_3d
                        target_grasp.grasp_width = fsm.last_good_target_grasp.grasp_width
                        target_grasp.grasp_axis_3d = fsm.last_good_target_grasp.grasp_axis_3d

        return last_mask, False, target_grasp
