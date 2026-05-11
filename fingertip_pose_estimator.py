import numpy as np
from enum import Enum
from stretch4_gripper_modeling_and_control import gripper_camera as gc

class FingertipEstimationMethod(Enum):
    VISUAL_BOTH = 1
    PREVIOUS_ESTIMATOR_BOTH = 2
    KINEMATIC_BOTH = 3
    VISUAL_LEFT_KINEMATIC_MIRROR_RIGHT = 4
    VISUAL_RIGHT_KINEMATIC_MIRROR_LEFT = 5

class FingertipPoseEstimator:
    def __init__(self, visualizer, detector, mirror_config=None, max_history_frames=10):
        self.visualizer = visualizer
        self.detector = detector
        self.mirror_config = mirror_config
        self.recent_fingertip_centers_3d = []
        self.max_history_frames = max_history_frames

    def process(self, color_image, rgb_camera_info, closest_joint_state, reconstructor):
        vis_fingertips = None
        fingertip_center_3d = None
        fingertip_center_2d = None
        estimation_method = None
        active_fingertips_for_display = None
        matched_pct = None

        if self.visualizer is not None and closest_joint_state is not None:
            matched_pct = closest_joint_state['gripper']['pos_pct']
            vis_fingertips = self.detector.process_image(color_image, rgb_camera_info, pos_pct=matched_pct)
            
            active_fingertips = {}
            if vis_fingertips is not None:
                if 'left' in vis_fingertips and 'right' in vis_fingertips:
                    estimation_method = FingertipEstimationMethod.VISUAL_BOTH
                    active_fingertips['left'] = vis_fingertips['left']
                    active_fingertips['right'] = vis_fingertips['right']
                elif 'left' in vis_fingertips and self.mirror_config is not None:
                    estimation_method = FingertipEstimationMethod.VISUAL_LEFT_KINEMATIC_MIRROR_RIGHT
                    active_fingertips['left'] = vis_fingertips['left']
                    M_G = np.array(self.mirror_config['left_to_right']['M_G'])
                    M_L = np.array(self.mirror_config['left_to_right']['M_L'])
                    T_L = np.eye(4)
                    T_L[:3, :3] = np.column_stack((vis_fingertips['left']['x_axis'], vis_fingertips['left']['y_axis'], vis_fingertips['left']['z_axis']))
                    T_L[:3, 3] = vis_fingertips['left']['pos']
                    T_R = M_G @ T_L @ M_L
                    active_fingertips['right'] = {
                        'pos': T_R[:3, 3], 'x_axis': T_R[:3, 0], 'y_axis': T_R[:3, 1], 'z_axis': T_R[:3, 2]
                    }
                elif 'right' in vis_fingertips and self.mirror_config is not None:
                    estimation_method = FingertipEstimationMethod.VISUAL_RIGHT_KINEMATIC_MIRROR_LEFT
                    active_fingertips['right'] = vis_fingertips['right']
                    M_G = np.array(self.mirror_config['right_to_left']['M_G'])
                    M_L = np.array(self.mirror_config['right_to_left']['M_L'])
                    T_R = np.eye(4)
                    T_R[:3, :3] = np.column_stack((vis_fingertips['right']['x_axis'], vis_fingertips['right']['y_axis'], vis_fingertips['right']['z_axis']))
                    T_R[:3, 3] = vis_fingertips['right']['pos']
                    T_L = M_G @ T_R @ M_L
                    active_fingertips['left'] = {
                        'pos': T_L[:3, 3], 'x_axis': T_L[:3, 0], 'y_axis': T_L[:3, 1], 'z_axis': T_L[:3, 2]
                    }
                elif len(self.recent_fingertip_centers_3d) > 0:
                    estimation_method = FingertipEstimationMethod.PREVIOUS_ESTIMATOR_BOTH
                    fingertip_center_3d = self.recent_fingertip_centers_3d[-1]
                    self.recent_fingertip_centers_3d.pop(0)
                else:
                    estimation_method = FingertipEstimationMethod.KINEMATIC_BOTH
            else:
                if len(self.recent_fingertip_centers_3d) > 0:
                    estimation_method = FingertipEstimationMethod.PREVIOUS_ESTIMATOR_BOTH
                    fingertip_center_3d = self.recent_fingertip_centers_3d[-1]
                    self.recent_fingertip_centers_3d.pop(0)
                else:
                    estimation_method = FingertipEstimationMethod.KINEMATIC_BOTH
                    
            if estimation_method in [FingertipEstimationMethod.VISUAL_BOTH, FingertipEstimationMethod.VISUAL_LEFT_KINEMATIC_MIRROR_RIGHT, FingertipEstimationMethod.VISUAL_RIGHT_KINEMATIC_MIRROR_LEFT]:
                fingertip_center_3d = (active_fingertips['left']['pos'] + active_fingertips['right']['pos']) / 2.0
                self.recent_fingertip_centers_3d.append(fingertip_center_3d)
                if len(self.recent_fingertip_centers_3d) > self.max_history_frames:
                    self.recent_fingertip_centers_3d.pop(0)
                active_fingertips_for_display = active_fingertips

            if estimation_method == FingertipEstimationMethod.KINEMATIC_BOTH:
                full_history_list = reconstructor.get_history_list()
                direction = 'closing'
                if len(full_history_list) > 10:
                    recent = full_history_list[-10:]
                    diff = recent[-1]['gripper']['pos_pct'] - recent[0]['gripper']['pos_pct']
                    direction = 'opening' if diff > 0 else 'closing'
                
                pos_pred_l, _ = self.visualizer.predict('left', matched_pct, direction)
                pos_pred_r, _ = self.visualizer.predict('right', matched_pct, direction)
                if pos_pred_l is not None and pos_pred_r is not None:
                    fingertip_center_3d = (pos_pred_l + pos_pred_r) / 2.0
                    active_fingertips_for_display = {
                        'left': {'pos': pos_pred_l},
                        'right': {'pos': pos_pred_r}
                    }

            if fingertip_center_3d is not None:
                fingertip_center_2d = gc.pixel_from_3d(fingertip_center_3d, rgb_camera_info)

        return {
            'fingertip_center_3d': fingertip_center_3d,
            'fingertip_center_2d': fingertip_center_2d,
            'active_fingertips_for_display': active_fingertips_for_display,
            'estimation_method': estimation_method,
            'vis_fingertips': vis_fingertips,
            'matched_pct': matched_pct
        }
