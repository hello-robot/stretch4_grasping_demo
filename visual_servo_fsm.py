import time
from enum import Enum
import numpy as np
import visual_servo_fsm_params as vp
from perception_utils import compute_look_at_rot_velocities, get_median_object_depth
from perception_utils import compute_3d_look_at_rot_velocities, compute_3d_align_rot_velocities
from stretch4_gripper_modeling_and_control import gripper_camera as gc
import sys

class State(Enum):
    INITIALIZE = 0
    FIND_OBJECT = 1
    MOVE_TO_OBJECT_HEIGHT = 2
    MOVE_TO_FRONT_OF_OBJECT = 13
    PRE_APPROACH_LOOK = 6
    APPROACH_OBJECT = 3
    LOOK_AT_OBJECT = 4
    ALIGN_GRASP = 7
    PRE_GRASP = 8
    APPROACH_GRASP = 9
    EXECUTE_GRASP = 10
    LIFT_OBJECT = 11
    RETRACT_GRASP = 12
    LOST_OBJECT = 5

class BaseState:
    @property
    def window_name(self):
        return "Camera Stream"

    def on_enter(self, fsm):
        pass

    def update(self, fsm, context):
        """
        context is a dictionary containing all required data from the main loop.
        Returns next_state, dict_of_stats_to_update_in_main_loop
        """
        return None, {}

    def on_exit(self, fsm):
        pass

class InitializeState(BaseState):
    def update(self, fsm, context):
        closest_joint_state = context.get('closest_joint_state')
        display_image = context.get('display_image')
        import cv2

        if closest_joint_state is not None and closest_joint_state['gripper']['pos_pct'] >= vp.INITIALIZE_TARGET_POS_PCT:
            return State.FIND_OBJECT, {'display_image': display_image}
        else:
            cmd = {
                'v_desired': [0.0, 0.0, 0.0],
                'rot_change': [0.0, 0.0, 0.0],
                'grip': {'pos_pct': 100.0, 'speed': vp.INITIALIZE_SPEED, 'accel': vp.INITIALIZE_ACCEL},
                'control_mode': 3
            }
            fsm.cmd_socket.send_pyobj(cmd)
            
        cv2.putText(display_image, "State: INITIALIZE... Opening Gripper", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return None, {'display_image': display_image}

class FindObjectState(BaseState):
    @property
    def window_name(self):
        return "Molmo Object Initialization"

    def update(self, fsm, context):
        rgb_image = context['rgb_image']
        scaled_color_image = context['scaled_color_image']
        ui_manager = context['ui_manager']
        perception_pipeline = context['perception_pipeline']
        tracker = context['tracker']
        
        target_point, pt_scaled, generated_text = perception_pipeline.run_molmo_detection(rgb_image, fsm.args.object_description)
        
        next_state = None
        if target_point is not None:
            print(f"Found object at target point: {target_point}. Initializing SAM 2 tracking...")
            tracker.initialize_tracking(rgb_image, target_point)
            next_state = State.LOOK_AT_OBJECT
            
        molmo_display_image = ui_manager.create_molmo_ui(scaled_color_image, generated_text, fsm.args.object_description, pt_scaled, fsm.current_state.name)
        return next_state, {'molmo_display_image': molmo_display_image}

class TrackingState(BaseState):
    """
    Base class for all states that require SAM2 tracking. 
    Handles updating the tracker and fetching the target grasp, reducing redundancy.
    """
    @property
    def window_name(self):
        return "Visual Servoing Output"

    def update(self, fsm, context):
        perception_pipeline = context['perception_pipeline']
        ui_manager = context['ui_manager']
        rgb_camera_info = context.get('rgb_camera_info')
        # Re-extract scaled camera info properly since ui_manager expects it
        import visualization_utils as vu
        _, scaled_camera_info = vu.apply_display_scale(context['rgb_image'], fsm.args.display_scale, camera_info=rgb_camera_info)

        last_mask, target_lost, target_grasp = perception_pipeline.update_tracking_and_grasp(
            fsm, context['rgb_image'], context['depth_image'], rgb_camera_info, context['H'], context['W']
        )

        if target_lost:
            fsm.handle_tracking_lost()
            return None, {'target_lost': True}

        # Subclasses execute their specific logic
        next_state, centroid, median_depth, v_desired, rot_change = self.tracking_step(
            fsm, context, last_mask, target_grasp
        )

        display_image = ui_manager.overlay_tracking(
            context['scaled_color_image'], last_mask, centroid, fsm.current_state.name, 
            median_depth, v_desired, rot_change, target_grasp, scaled_camera_info, fsm.args.grasp_estimator
        )

        return next_state, {'display_image': display_image}

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        return None, None, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

class MoveToObjectHeightState(TrackingState):
    def on_enter(self, fsm):
        self.height_settle_frames = 0

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        rgb_camera_info = context.get('rgb_camera_info')
        closest_joint_state = context.get('closest_joint_state')
        curr_wrist_roll = context.get('curr_wrist_roll')
        curr_wrist_pitch = context.get('curr_wrist_pitch')
        fingertip_center_3d = context.get('fingertip_center_3d')
        
        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)
        elif median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)

        # 1. move wrist pitch to 0.0
        pitch_cmd = 0.0
        if curr_wrist_pitch is not None:
            # The sign is flipped because positive velocity commands rotate the pitch upwards, 
            # but the joint position might be positive when looking downwards.
            pitch_cmd = vp.KP_PITCH * curr_wrist_pitch
            pitch_cmd = max(min(pitch_cmd, 1.0), -1.0)
            
        # 2. move wrist roll to 0.0
        roll_cmd = 0.0
        if curr_wrist_roll is not None:
            # Flipped sign to match joint velocity convention vs position
            roll_cmd = vp.KP_ROLL * curr_wrist_roll
            roll_cmd = max(min(roll_cmd, 1.0), -1.0)

        # 3. move wrist yaw to keep the 2D target grasp location horizontally centered
        yaw_cmd = 0.0
        if object_center_3d is not None and fingertip_center_3d is not None:
            error_x = object_center_3d[0] - fingertip_center_3d[0]
            depth_val = max(object_center_3d[2], vp.MIN_DEPTH_FOR_YAW_M)
            norm_err_x = error_x / depth_val
            yaw_cmd = -vp.KP_YAW * norm_err_x
            yaw_cmd = max(min(yaw_cmd, 1.0), -1.0)
            
        rot_change = [yaw_cmd, pitch_cmd, roll_cmd]
        rot_change = [r * vp.PRECISE_ALIGNMENT_ROT_DAMPING for r in rot_change]

        # 4. change the height of the gripper using Mode #2 control
        v_desired = [0.0, 0.0, 0.0]
        z_error_m = 0.0
        if object_center_3d is not None and fingertip_center_3d is not None:
            error_y = object_center_3d[1] - fingertip_center_3d[1]
            z_error_m = -error_y  # negative error_y means we should go up
            vz = vp.KP_HEIGHT * z_error_m
            vz = max(min(vz, vp.MAX_HEIGHT_SPEED_M_S), -vp.MAX_HEIGHT_SPEED_M_S)
            v_desired[2] = vz

        next_state = None
        if curr_wrist_pitch is not None and curr_wrist_roll is not None and object_center_3d is not None and fingertip_center_3d is not None:
            if abs(curr_wrist_pitch) < vp.PITCH_TOLERANCE_RAD and \
               abs(curr_wrist_roll) < vp.ROLL_TOLERANCE_RAD and \
               abs(yaw_cmd) < vp.LOOK_AT_ROT_VEL_TOLERANCE and \
               abs(z_error_m) < vp.HEIGHT_TOLERANCE_M:
                self.height_settle_frames += 1
                if self.height_settle_frames >= vp.HEIGHT_SETTLE_FRAMES:
                    fsm.has_aligned_height = True
                    if getattr(fsm.args, 'move_to_front', False):
                        next_state = State.MOVE_TO_FRONT_OF_OBJECT
                    else:
                        next_state = State.APPROACH_OBJECT
            else:
                self.height_settle_frames = 0

        cmd = {
            'v_desired': v_desired,
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 2
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, v_desired, rot_change

class PreApproachLookState(TrackingState):
    def on_enter(self, fsm):
        self.look_at_settle_frames = 0

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)
        elif median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)

        visualizer = context.get('visualizer')
        plane_normal = visualizer.normal if visualizer is not None else None
        curr_wrist_roll = context.get('curr_wrist_roll')
        rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)
        
        rot_change = [r * vp.PRE_APPROACH_LOOK_ROT_DAMPING for r in rot_change]

        if np.linalg.norm(rot_change[:2]) < vp.LOOK_AT_ROT_VEL_TOLERANCE:
            self.look_at_settle_frames += 1
        else:
            self.look_at_settle_frames = 0
        
        next_state = None
        if self.look_at_settle_frames >= vp.LOOK_AT_SETTLE_FRAMES:
            next_state = State.APPROACH_OBJECT

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)
        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change


class MoveToFrontOfObjectState(TrackingState):
    def on_enter(self, fsm):
        self.stage = 'probe' # 'probe', 'move', 'done'
        self.explore_dir = 1.0 # 1.0 for left (+y), -1.0 for right (-y)
        self.min_width = float('inf')
        self.last_width = None
        self.settle_frames = 0
        self.probe_frames = 0
        self.move_frames = 0
        self.start_width = None
        self.minimum_move_frames = vp.FRONT_MIN_MOVE_FRAMES
        
    def _estimate_width(self, last_mask, target_grasp, context):
        if target_grasp is not None:
            return target_grasp.grasp_width
        return None

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        rgb_camera_info = context.get('rgb_camera_info')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)
        elif median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)

        yaw_cmd = 0.0
        if centroid is not None:
            error_x = (centroid[0] - (w / 2.0)) / w
            yaw_cmd = -vp.KP_YAW * error_x
            yaw_cmd = max(min(yaw_cmd, 1.0), -1.0)
            
        rot_change = [yaw_cmd * vp.PRECISE_ALIGNMENT_ROT_DAMPING, 0.0, 0.0]
        
        current_width = self._estimate_width(last_mask, target_grasp, context)
        
        v_desired = [0.0, 0.0, 0.0]
        next_state = None
        
        if current_width is not None:
            if self.last_width is None:
                self.last_width = current_width
                self.start_width = current_width
                self.min_width = current_width
                
            if current_width < self.min_width:
                self.min_width = current_width
                
            if getattr(fsm.args, 'verbose', False):
                print(f"MOVE_TO_FRONT_OF_OBJECT: Moving in direction {self.explore_dir}, current width {current_width}, minimum width {self.min_width}, stage {self.stage}, move_frames {self.move_frames}, probe_frames {self.probe_frames}, minimum_move_frames {self.minimum_move_frames}")
            
            if self.stage == 'move':
                v_desired[1] = self.explore_dir * vp.FRONT_EXPLORE_SPEED_M_S
                self.move_frames += 1
                if self.move_frames > self.minimum_move_frames and (current_width > self.min_width + vp.FRONT_WIDTH_TOLERANCE_M):
                    self.stage = 'done'
            
            elif self.stage == 'probe':
                v_desired[1] = self.explore_dir * vp.FRONT_EXPLORE_SPEED_M_S
                self.probe_frames += 1

                if self.probe_frames > vp.FRONT_MIN_PROBE_FRAMES:
                    if current_width > self.start_width + vp.FRONT_WIDTH_PROBE_TOLERANCE_M:
                        # moving in the wrong direction
                        if getattr(fsm.args, 'verbose', False):
                            print("MOVE_TO_FRONT_OF_OBJECT: Moving in the wrong direction. Reversing direction.")
                        self.explore_dir *= -1.0
                        # reset the search for the minimum width
                        self.min_width = current_width
                        self.stage = 'move'
                        self.minimum_move_frames = vp.FRONT_MIN_MOVE_FRAMES
                    elif current_width < self.start_width - vp.FRONT_WIDTH_PROBE_TOLERANCE_M:
                        # moving in the correct direction
                        if getattr(fsm.args, 'verbose', False):
                            print("MOVE_TO_FRONT_OF_OBJECT: Moving in the correct direction.")
                        self.stage = 'move'
                        self.minimum_move_frames = 0
            
            elif self.stage == 'done':
                v_desired[1] = 0.0
                self.settle_frames += 1
                if self.settle_frames >= vp.FRONT_SETTLE_FRAMES:
                    next_state = State.APPROACH_OBJECT
                    
            self.last_width = current_width
        else:
            v_desired[1] = 0.0

        cmd = {
            'v_desired': v_desired,
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 2
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, v_desired, rot_change

class ApproachObjectState(TrackingState):
    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)

        object_center_3d = None
        if median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)

        visualizer = context.get('visualizer')
        plane_normal = visualizer.normal if visualizer is not None else None
        curr_wrist_roll = context.get('curr_wrist_roll')
        rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)

        dist = None
        if object_center_3d is not None and fingertip_center_3d is not None:
            dist = np.linalg.norm(object_center_3d - fingertip_center_3d)
        elif median_depth is not None:
            dist = median_depth
            
        v_desired = [0.0, 0.0, 0.0]
        next_state = None
        if dist is not None and dist <= vp.APPROACH_OBJECT_TARGET_DIST_M:
            print(f"Reached target Euclidean distance of {dist:.2f}m.")
            next_state = State.LOOK_AT_OBJECT
        else:
            v_desired = [vp.APPROACH_OBJECT_FORWARD_SPEED, 0.0, 0.0]

        cmd = {
            'v_desired': v_desired,
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)
        return next_state, centroid, median_depth, v_desired, rot_change

class LookAtObjectState(TrackingState):
    def on_enter(self, fsm):
        self.look_at_settle_frames = 0

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if median_depth is not None:
            object_center_3d = gc.pixel_to_3d(centroid, median_depth, rgb_camera_info)
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)

        visualizer = context.get('visualizer')
        plane_normal = visualizer.normal if visualizer is not None else None
        curr_wrist_roll = context.get('curr_wrist_roll')
        rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)
        rot_change = [r * vp.PRECISE_ALIGNMENT_ROT_DAMPING for r in rot_change]

        if np.linalg.norm(rot_change[:2]) < vp.LOOK_AT_ROT_VEL_TOLERANCE:
            self.look_at_settle_frames += 1
        else:
            self.look_at_settle_frames = 0
            
        next_state = None
        if self.look_at_settle_frames >= vp.LOOK_AT_SETTLE_FRAMES:
            if not getattr(fsm, 'has_aligned_height', False):
                next_state = State.MOVE_TO_OBJECT_HEIGHT
            elif target_grasp is not None:
                next_state = State.ALIGN_GRASP
            elif fsm.args.grasp_estimator == 'none':
                print("No grasp estimator specified. LOOK_AT_OBJECT completed. Ending servoing sequence.")
                fsm.send_zero_command()
                sys.exit(0)

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)
        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change

class AlignGraspState(TrackingState):
    def on_enter(self, fsm):
        self.align_settle_frames = 0

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')
        curr_wrist_roll = context.get('curr_wrist_roll')
        active_fingertips = context.get('active_fingertips')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)

        angle_err = 0.0
        if target_grasp is not None and active_fingertips is not None:
            if 'left' in active_fingertips and 'right' in active_fingertips:
                fingertip_axis = active_fingertips['right']['pos'] - active_fingertips['left']['pos']
                visualizer = context.get('visualizer')
                plane_normal = visualizer.normal if visualizer is not None else None
                rot_change, angle_err = compute_3d_align_rot_velocities(
                    
                    fingertip_center_3d, object_center_3d, plane_normal, target_grasp.grasp_axis_3d, fingertip_axis, 
                    curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH, vp.KP_ROLL
                )
            else:
                visualizer = context.get('visualizer')
                plane_normal = visualizer.normal if visualizer is not None else None
                curr_wrist_roll = context.get('curr_wrist_roll')
                rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)
        else:
            visualizer = context.get('visualizer')
            plane_normal = visualizer.normal if visualizer is not None else None
            curr_wrist_roll = context.get('curr_wrist_roll')
            rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)
            
        rot_change = [r * vp.PRECISE_ALIGNMENT_ROT_DAMPING for r in rot_change]
        rot_change[0] *= vp.ALIGN_GRASP_YAW_DAMPING_MULTIPLIER

        j_pos_cmds = None
        next_state = None
        if target_grasp is not None and active_fingertips is not None and 'left' in active_fingertips and 'right' in active_fingertips:
            target_wrist_roll = curr_wrist_roll - angle_err
            j_pos_cmds = {'wrist_roll': target_wrist_roll}
            
            if abs(angle_err) < vp.ALIGN_ANGLE_TOLERANCE_RAD and np.linalg.norm(rot_change[:2]) < vp.LOOK_AT_ROT_VEL_TOLERANCE:
                self.align_settle_frames += 1
                if self.align_settle_frames >= vp.ALIGN_SETTLE_FRAMES:
                    next_state = State.PRE_GRASP
            else:
                self.align_settle_frames = 0

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 1
        }
        if j_pos_cmds is not None:
            cmd['joint_position_commands'] = j_pos_cmds
        fsm.cmd_socket.send_pyobj(cmd)
        
        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change

class PreGraspState(TrackingState):
    def on_enter(self, fsm):
        self.pre_grasp_settle_frames = 0

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')
        curr_wrist_roll = context.get('curr_wrist_roll')
        active_fingertips = context.get('active_fingertips')
        visualizer = context.get('visualizer')
        closest_joint_state = context.get('closest_joint_state')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)

        if target_grasp is not None and active_fingertips is not None and 'left' in active_fingertips and 'right' in active_fingertips:
            fingertip_axis = active_fingertips['right']['pos'] - active_fingertips['left']['pos']
            visualizer = context.get('visualizer')
            plane_normal = visualizer.normal if visualizer is not None else None
            rot_change, _ = compute_3d_align_rot_velocities(
                
                fingertip_center_3d, object_center_3d, plane_normal, target_grasp.grasp_axis_3d, fingertip_axis, 
                curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH, vp.KP_ROLL
            )
        else:
            visualizer = context.get('visualizer')
            plane_normal = visualizer.normal if visualizer is not None else None
            curr_wrist_roll = context.get('curr_wrist_roll')
            rot_change = compute_3d_look_at_rot_velocities(fingertip_center_3d, object_center_3d, plane_normal, curr_wrist_roll, vp.KP_YAW, vp.KP_PITCH)
            
        rot_change = [r * vp.PRECISE_ALIGNMENT_ROT_DAMPING for r in rot_change]

        grip_cmd = None
        next_state = None
        if target_grasp is not None and visualizer is not None and closest_joint_state is not None:
            target_pre_grasp_width = target_grasp.grasp_width + vp.PRE_GRASP_WIDTH_MARGIN_M
            target_pre_pos_pct = fsm.calculate_pos_pct_for_aperture(visualizer, target_pre_grasp_width)

            current_pct = closest_joint_state['gripper']['pos_pct']
            diff = current_pct - target_pre_pos_pct
            
            grip_cmd = {'pos_pct': target_pre_pos_pct, 'speed': vp.PRE_GRASP_FF_SPEED, 'accel': vp.PRE_GRASP_FF_ACCEL}
            
            if abs(diff) <= vp.PRE_GRASP_TOLERANCE_PCT and np.linalg.norm(rot_change[:2]) < vp.LOOK_AT_ROT_VEL_TOLERANCE:
                self.pre_grasp_settle_frames += 1
                if self.pre_grasp_settle_frames >= vp.PRE_GRASP_SETTLE_FRAMES:
                    next_state = State.APPROACH_GRASP
            else:
                self.pre_grasp_settle_frames = 0

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': grip_cmd,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change

class ApproachGraspState(TrackingState):
    def on_enter(self, fsm):
        fsm.initial_z_dist = None

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        fingertip_center_3d = context.get('fingertip_center_3d')
        rgb_camera_info = context.get('rgb_camera_info')
        curr_wrist_roll = context.get('curr_wrist_roll')
        active_fingertips = context.get('active_fingertips')
        visualizer = context.get('visualizer')
        closest_joint_state = context.get('closest_joint_state')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        
        object_center_3d = None
        if target_grasp is not None:
            object_center_3d = target_grasp.center_3d
            centroid = gc.pixel_from_3d(object_center_3d, rgb_camera_info)

        # During APPROACH_GRASP, we should not adjust any wrist angles (roll, pitch, or yaw).
        # We assume the alignment from ALIGN_GRASP is sufficient, and we purely move forward.
        rot_change = [0.0, 0.0, 0.0]

        grip_cmd = None
        v_desired = [0.0, 0.0, 0.0]
        next_state = None
        
        if target_grasp is not None and visualizer is not None and closest_joint_state is not None:
            target_pre_grasp_width = target_grasp.grasp_width + vp.PRE_GRASP_WIDTH_MARGIN_M
            target_pre_pos_pct = fsm.calculate_pos_pct_for_aperture(visualizer, target_pre_grasp_width)
            current_pct = closest_joint_state['gripper']['pos_pct']
            diff = current_pct - target_pre_pos_pct
            
            if abs(diff) > vp.APERTURE_FEEDBACK_DEADBAND_PCT:
                disp = np.clip(-diff, -vp.PRE_GRASP_MAX_DISP, vp.PRE_GRASP_MAX_DISP)
                grip_cmd = {'pos_pct_disp': float(disp), 'speed': vp.PRE_GRASP_SPEED, 'accel': vp.PRE_GRASP_ACCEL}

            if fingertip_center_3d is not None:
                if active_fingertips is not None and 'left' in active_fingertips and 'right' in active_fingertips:
                    left_ft_z = active_fingertips['left']['pos'][2]
                    right_ft_z = active_fingertips['right']['pos'][2]
                else:
                    left_ft_z = fingertip_center_3d[2]
                    right_ft_z = fingertip_center_3d[2]

                z_dist = target_grasp.left_contact_3d[2] - left_ft_z
                
                if fsm.initial_z_dist is None:
                    fsm.initial_z_dist = z_dist

                if z_dist <= vp.APPROACH_GRASP_SLOW_DIST_M or fsm.initial_z_dist <= vp.APPROACH_GRASP_SLOW_DIST_M:
                    fwd_speed = vp.APPROACH_GRASP_SLOW_SPEED
                else:
                    fraction = (z_dist - vp.APPROACH_GRASP_SLOW_DIST_M) / (fsm.initial_z_dist - vp.APPROACH_GRASP_SLOW_DIST_M)
                    fraction = np.clip(fraction, 0.0, 1.0)
                    fwd_speed = vp.APPROACH_GRASP_SLOW_SPEED + fraction * (vp.APPROACH_GRASP_FAST_SPEED - vp.APPROACH_GRASP_SLOW_SPEED)
                    
                v_desired = [fwd_speed, 0.0, 0.0]

                if (left_ft_z > target_grasp.left_contact_3d[2] + vp.APPROACH_GRASP_DEPTH_MARGIN_M) and (right_ft_z > target_grasp.right_contact_3d[2] + vp.APPROACH_GRASP_DEPTH_MARGIN_M):
                    fsm.exec_grasp_width = target_grasp.grasp_width
                    fsm.target_close_pos_pct = vp.DEFAULT_CLOSE_POS_PCT
                    next_state = State.EXECUTE_GRASP

        cmd = {
            'v_desired': v_desired,
            'rot_change': rot_change,
            'grip': grip_cmd,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, v_desired, rot_change

class ExecuteGraspState(TrackingState):
    def on_enter(self, fsm):
        self.exec_last_pct = None
        self.exec_stall_time = None

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        rgb_camera_info = context.get('rgb_camera_info')
        closest_joint_state = context.get('closest_joint_state')

        active_fingertips = context.get('active_fingertips')
        visualizer = context.get('visualizer')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        if target_grasp is not None:
            centroid = gc.pixel_from_3d(target_grasp.center_3d, rgb_camera_info)

        rot_change = [0.0, 0.0, 0.0]
        grip_cmd = None
        next_state = None
        
        if closest_joint_state is not None:
            current_pct = closest_joint_state['gripper']['pos_pct']
            diff = current_pct - getattr(fsm, 'target_close_pos_pct', vp.DEFAULT_CLOSE_POS_PCT)
            
            if diff > vp.EXECUTE_GRASP_DEADBAND:
                disp = np.clip(-diff, -vp.EXECUTE_GRASP_MAX_DISP, vp.EXECUTE_GRASP_MAX_DISP)
                grip_cmd = {'pos_pct_disp': float(disp), 'speed': vp.EXECUTE_GRASP_SPEED, 'accel': vp.EXECUTE_GRASP_ACCEL}
            
            # Deflection force control check
            deflection_target_met = False
            if getattr(fsm, 'exec_grasp_width', 0) >= vp.MIN_VALID_GRASP_WIDTH_M:
                if active_fingertips is not None and 'left' in active_fingertips and 'right' in active_fingertips and visualizer is not None:
                    # 1. Visual Aperture
                    left_pos_vis = active_fingertips['left']['pos']
                    right_pos_vis = active_fingertips['right']['pos']
                    visual_aperture = np.linalg.norm(left_pos_vis - right_pos_vis)
                    
                    # 2. Kinematic Aperture
                    left_pos_kin, _ = visualizer.predict('left', current_pct, 'closing')
                    right_pos_kin, _ = visualizer.predict('right', current_pct, 'closing')
                    if left_pos_kin is not None and right_pos_kin is not None:
                        kinematic_aperture = np.linalg.norm(left_pos_kin - right_pos_kin)
                        
                        # 3. Deflection
                        deflection = visual_aperture - kinematic_aperture
                        
                        # 4. Check target
                        if deflection >= vp.EXECUTE_GRASP_TARGET_DEFLECTION_M:
                            deflection_target_met = True

            if deflection_target_met:
                next_state = State.LIFT_OBJECT
            else:
                # Fallback stall check (for small objects or if tracking fails)
                if self.exec_last_pct is None:
                    self.exec_last_pct = current_pct
                    self.exec_stall_time = time.time()
                
                if abs(current_pct - self.exec_last_pct) > vp.EXEC_STALL_DIFF_PCT:
                    self.exec_last_pct = current_pct
                    self.exec_stall_time = time.time()
                elif time.time() - self.exec_stall_time > vp.EXEC_STALL_TIME_S:
                    next_state = State.LIFT_OBJECT

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': grip_cmd,
            'control_mode': 1
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change

class LiftObjectState(TrackingState):
    def on_enter(self, fsm):
        self.lift_target_pos = None

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        rgb_camera_info = context.get('rgb_camera_info')
        closest_joint_state = context.get('closest_joint_state')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        if target_grasp is not None:
            centroid = gc.pixel_from_3d(target_grasp.center_3d, rgb_camera_info)

        rot_change = [0.0, 0.0, 0.0]
        j_pos_cmds = None
        next_state = None

        if self.lift_target_pos is None and closest_joint_state is not None:
            desired_height = closest_joint_state['lift']['height'] + vp.LIFT_OBJECT_DIST_M
            self.lift_target_pos = min(desired_height, vp.MAX_LIFT_HEIGHT_M)

        if self.lift_target_pos is not None and closest_joint_state is not None:
            j_pos_cmds = {'lift': self.lift_target_pos}
            if closest_joint_state['lift']['height'] >= self.lift_target_pos - vp.LIFT_HEIGHT_TOLERANCE_M:
                next_state = State.RETRACT_GRASP

        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': rot_change,
            'grip': None,
            'control_mode': 3
        }
        if j_pos_cmds is not None:
            cmd['joint_position_commands'] = j_pos_cmds
            
        fsm.cmd_socket.send_pyobj(cmd)

        return next_state, centroid, median_depth, [0.0, 0.0, 0.0], rot_change

class RetractGraspState(TrackingState):
    def on_enter(self, fsm):
        self.retract_start_time = time.time()

    def tracking_step(self, fsm, context, last_mask, target_grasp):
        w, h = context['W'], context['H']
        depth_image = context.get('depth_image')
        rgb_camera_info = context.get('rgb_camera_info')

        _, centroid = compute_look_at_rot_velocities(last_mask, w, h, vp.KP_YAW, vp.KP_PITCH, image_servo_mode='image_center_to_object')
        median_depth = get_median_object_depth(last_mask, depth_image)
        if target_grasp is not None:
            centroid = gc.pixel_from_3d(target_grasp.center_3d, rgb_camera_info)

        if time.time() - self.retract_start_time > vp.RETRACT_TIME_S:
            print("Retraction phase completed. Ending visual servoing sequence.")
            fsm.send_zero_command()
            sys.exit(0)
            
        v_desired = [-vp.RETRACT_BACK_SPEED_M_S, 0.0, 0.0]

        cmd = {
            'v_desired': v_desired,
            'rot_change': [0.0, 0.0, 0.0],
            'grip': None,
            'control_mode': 2
        }
        fsm.cmd_socket.send_pyobj(cmd)

        return None, centroid, median_depth, v_desired, [0.0, 0.0, 0.0]

class LostObjectState(BaseState):
    def update(self, fsm, context):
        return None, {}

class VisualServoFSM:
    def __init__(self, cmd_socket, args):
        self.cmd_socket = cmd_socket
        self.args = args
        self.state_instances = {
            State.INITIALIZE: InitializeState(),
            State.FIND_OBJECT: FindObjectState(),
            State.MOVE_TO_OBJECT_HEIGHT: MoveToObjectHeightState(),
            State.MOVE_TO_FRONT_OF_OBJECT: MoveToFrontOfObjectState(),
            State.PRE_APPROACH_LOOK: PreApproachLookState(),
            State.APPROACH_OBJECT: ApproachObjectState(),
            State.LOOK_AT_OBJECT: LookAtObjectState(),
            State.ALIGN_GRASP: AlignGraspState(),
            State.PRE_GRASP: PreGraspState(),
            State.APPROACH_GRASP: ApproachGraspState(),
            State.EXECUTE_GRASP: ExecuteGraspState(),
            State.LIFT_OBJECT: LiftObjectState(),
            State.RETRACT_GRASP: RetractGraspState(),
            State.LOST_OBJECT: LostObjectState(),
        }
        self.current_state = State.INITIALIZE
        self.has_aligned_height = False
        self.state_instances[self.current_state].on_enter(self)
        self.state_start_time = time.time()
        print(f"FSM: {self.current_state.name} beginning first iteration.")
        
    def transition_to(self, new_state):
        print(f"FSM: Transitioning from {self.current_state.name} to {new_state.name}.")
        self.state_instances[self.current_state].on_exit(self)
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_instances[self.current_state].on_enter(self)
        print(f"FSM: {self.current_state.name} beginning first iteration.")
        
    def send_zero_command(self):
        cmd = {
            'v_desired': [0.0, 0.0, 0.0],
            'rot_change': [0.0, 0.0, 0.0],
            'grip': None,
            'control_mode': 1
        }
        self.cmd_socket.send_pyobj(cmd)
        
    def handle_tracking_lost(self):
        print("SAM 2 target lost! Transitioning to State: LOST_OBJECT")
        self.transition_to(State.LOST_OBJECT)
        self.send_zero_command()
        print("Object was lost. Exiting the program.")
        
    def calculate_pos_pct_for_aperture(self, visualizer, target_aperture):
        best_pct = 0.0
        min_diff = float('inf')
        for pct in np.linspace(300, -100, 400):
            pos_l, _ = visualizer.predict('left', pct, 'closing')
            pos_r, _ = visualizer.predict('right', pct, 'closing')
            if pos_l is not None and pos_r is not None:
                dist = np.linalg.norm(pos_l - pos_r)
                diff = abs(dist - target_aperture)
                if diff < min_diff:
                    min_diff = diff
                    best_pct = pct
        return best_pct

    def update(self, context):
        """
        Delegates the update call to the current state, passing the context dictionary.
        Returns the stats dictionary populated by the state.
        """
        next_state, stats = self.state_instances[self.current_state].update(self, context)
        if next_state is not None:
            self.transition_to(next_state)
        return stats
