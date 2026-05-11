#!/usr/bin/env python3
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["QT_LOGGING_RULES"] = "*=false"
import argparse
import cv2
import zmq
import numpy as np

# Utilities
import vlm_utils as vlm
import sam2_utils as sam2
import visualization_utils as vu
from stretch4_gripper_modeling_and_control.visualization_utils import draw_predicted_frames
import visual_servo_fsm_params as vp
import shared_arguments
from stretch4_gripper_modeling_and_control import telemetry_utils as tu
from visual_servo_fsm import VisualServoFSM, State
from stretch4_gripper_modeling_and_control.swept_volume_model import SweptVolumeModel

try:
    from stretch4_gripper_modeling_and_control import gripper_networking as gn
except ImportError:
    pass

from stretch4_gripper_modeling_and_control.fingertip_visualizer import FingertipVisualizer

from stretch4_gripper_modeling_and_control.fingertip_detector import add_fingertip_detector_args, process_fingertip_detector_args
from stretch4_gripper_modeling_and_control import gripper_camera as gc
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af

# New Architected Components
from fingertip_pose_estimator import FingertipPoseEstimator, FingertipEstimationMethod
from swept_volume_ui import SweptVolumeProcessor
from perception_pipeline import PerceptionPipeline
from ui_manager import UIManager


def main():
    parser = argparse.ArgumentParser(
        prog='Visual Servo Gripper',
        description='Receives gripper camera images, visually servos towards an object matching the text description using Molmo 2 and SAM 2 tracking.'
    )
    parser.add_argument('--image_servo_mode', type=str, choices=['none', 'image_center_to_object', 'fingertip_to_object'], default='none', help='Fallback 2D servoing methods.')
    parser.add_argument('--overlay_estimation_frames', action='store_true', help='Visualize frames and suction cups used to estimate the 3D fingertip center.')
    parser.add_argument('--bg_visibility', type=float, default=0.6, help='Background visibility for non-segmented regions.')
    parser.add_argument('--fg_amplification', type=float, default=0.5, help='Amplifies brightness of the segmented foreground region.')
    parser.add_argument('--swept_volume_to_pos', nargs='?', const=0.0, type=float, default=None, help='Visualize and segment the volume swept out by the fingertips.')
    parser.add_argument('--swept_volume_sampling_method', type=str, choices=['pos_pct', 'arc_length'], default='pos_pct', help='Method to uniformly sample the swept volume.')
    parser.add_argument('--swept_volume_samples', type=int, default=30, help='Number of uniform samples for the swept volume.')
    parser.add_argument('--swept_volume_mesh_visibility', type=float, default=1.0, help='Visibility of the swept volume wiremesh.')
    parser.add_argument('--grasp_estimator', type=str, choices=['none', 'ellipsoid_2d', 'ellipsoid_3d', 'depth_profile'], default='ellipsoid_2d', help='Method to estimate target grasp from segmentation mask.')
    parser.add_argument('--move_to_front', action='store_true', help='Do not skip the MOVE_TO_FRONT_OF_OBJECT state.')
    shared_arguments.add_shared_arguments(parser)
    add_fingertip_detector_args(parser)
    vu.add_display_scale_argument(parser)
    args = parser.parse_args()

    # Setup Sockets
    sub_context = zmq.Context()
    socket = sub_context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    pub_context = zmq.Context()
    cmd_socket = pub_context.socket(zmq.PUB)
    cmd_socket.setsockopt(zmq.SNDHWM, 1)
    cmd_socket.setsockopt(zmq.RCVHWM, 1)
    
    if args.remote:
        img_address = 'tcp://' + gn.robot_ip + ':' + str(gn.gripper_and_joints_port)
        cmd_address = 'tcp://*:' + str(gn.gripper_cmd_port)
    else:
        img_address = 'tcp://127.0.0.1:' + str(gn.gripper_and_joints_port)
        cmd_address = 'tcp://127.0.0.1:' + str(gn.gripper_cmd_port)
        
    print(f"Connecting Sub to {img_address} ... Binding Pub to {cmd_address}")
    socket.connect(img_address)
    cmd_socket.bind(cmd_address)

    print("Waiting for first telemetry message to extract robot_id...")
    first_msg = socket.recv_pyobj()
    robot_id = first_msg.get('robot_id')
    if not robot_id:
        raise ValueError("The first telemetry message did not contain a 'robot_id'. Ensure send_gripper_images_and_joint_states.py is up to date.")
    print(f"Received robot_id '{robot_id}' from telemetry.")

    args = shared_arguments.resolve_model_path(args, robot_id)

    # Load SAM 2
    sam2_model, sam2_processor = sam2.load_sam2_model(args.sam2_model_id, args.tracking_mode)
    if sam2_model is None:
        return
    tracker = sam2.SAM2Tracker((sam2_model, sam2_processor), args.tracking_mode)

    # Load Molmo 2
    molmo_proc, molmo_model = vlm.load_molmo_model()
    if not molmo_model or not molmo_proc:
        return

    # Visualizer & Detector
    print(f"Loading Fingertip Visualizer with model: {args.model}")
    visualizer = FingertipVisualizer(args.model)
    detector = process_fingertip_detector_args(args)
    
    import yaml
    print(f"Loading mirror transforms from {args.mirror_path}")
    with open(args.mirror_path, 'r') as f:
        mirror_config = yaml.safe_load(f)

    # Grasp Estimator
    grasp_estimator = None
    if args.grasp_estimator == 'ellipsoid_2d':
        from grasp_estimators import EllipsoidGraspEstimator
        grasp_estimator = EllipsoidGraspEstimator(use_3d=False)
    elif args.grasp_estimator == 'ellipsoid_3d':
        from grasp_estimators import EllipsoidGraspEstimator
        grasp_estimator = EllipsoidGraspEstimator(use_3d=True)
    elif args.grasp_estimator == 'depth_profile':
        from grasp_estimators import DepthProfileGraspEstimator
        grasp_estimator = DepthProfileGraspEstimator()

    # Setup Managers and Pipelines
    fingertip_pose_estimator = FingertipPoseEstimator(visualizer, detector, mirror_config)
    swept_volume_processor = SweptVolumeProcessor(visualizer, args)
    perception_pipeline = PerceptionPipeline(tracker, grasp_estimator, molmo_proc, molmo_model, args)
    ui_manager = UIManager(args)

    # Initialize FSM Engine
    fsm = VisualServoFSM(cmd_socket, args)
    reconstructor = tu.JointStateHistory(maxlen=400, warn_on_discontinuity=False)

    print("====================================")
    print(f"Visual Servo Active. State: {fsm.current_state.name}")
    print("Receiving frames and joint states... Press 'q' or 'Esc' to quit.")
    print("====================================")
    
    molmo_display_image = None

    try:
        while True:
            output_dict = socket.recv_pyobj()
            if 'color_image_compressed' in output_dict:
                color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            else:
                color_image = output_dict['color_image']
                
            depth_image = output_dict.get('depth_image', None)
            joint_history = output_dict.get('joint_state_history', [])
            closest = output_dict.get('closest_joint_state', None)
            reconstructor.add_states(joint_history)

            rgb_camera_info = {
                'camera_matrix': output_dict.get('camera_matrix', np.eye(3)),
                'distortion_coefficients': output_dict.get('distortion_coefficients', np.zeros(5))
            }
            
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            H, W = rgb_image.shape[:2]

            # 1. Fingertip Pose Estimation
            fingertip_data = fingertip_pose_estimator.process(color_image, rgb_camera_info, closest, reconstructor)
            fingertip_center_3d = fingertip_data['fingertip_center_3d']
            fingertip_center_2d = fingertip_data['fingertip_center_2d']
            active_fingertips_for_display = fingertip_data['active_fingertips_for_display']
            estimation_method = fingertip_data['estimation_method']
            vis_fingertips = fingertip_data['vis_fingertips']
            matched_pct = fingertip_data['matched_pct']

            # 2. Perception: Swept Volume Depth Segmentation
            segmented_rgb = color_image.copy()
            if args.swept_volume_to_pos is not None:
                segmented_rgb = swept_volume_processor.segment_depth_image(color_image, depth_image, rgb_camera_info, vis_fingertips)

            # Display Scaling
            scaled_color_image, scaled_camera_info = vu.apply_display_scale(segmented_rgb, args.display_scale, camera_info=rgb_camera_info)
            display_image = scaled_color_image.copy()

            # 3. UI: Swept Volume Mesh Overlay
            if args.swept_volume_to_pos is not None and visualizer is not None and vis_fingertips is not None:
                display_image = swept_volume_processor.draw_swept_volume_overlay(display_image, vis_fingertips, scaled_camera_info)
            scaled_color_image = display_image.copy()

            # 4. FSM Logic
            curr_wrist_roll = None
            curr_wrist_pitch = None
            if closest is not None:
                roll_dict = closest.get('wrist_roll') or closest.get('joint_wrist_roll')
                if roll_dict is not None:
                    if 'angle' in roll_dict: curr_wrist_roll = roll_dict['angle']
                    elif 'pos' in roll_dict: curr_wrist_roll = roll_dict['pos']
                    
                pitch_dict = closest.get('wrist_pitch') or closest.get('joint_wrist_pitch')
                if pitch_dict is not None:
                    if 'angle' in pitch_dict: curr_wrist_pitch = pitch_dict['angle']
                    elif 'pos' in pitch_dict: curr_wrist_pitch = pitch_dict['pos']

            context = {
                'rgb_image': rgb_image,
                'depth_image': depth_image,
                'color_image': color_image,
                'scaled_color_image': scaled_color_image,
                'H': H, 'W': W,
                'rgb_camera_info': rgb_camera_info,
                'closest_joint_state': closest,
                'fingertip_center_3d': fingertip_center_3d,
                'fingertip_center_2d': fingertip_center_2d,
                'active_fingertips': active_fingertips_for_display,
                'curr_wrist_roll': curr_wrist_roll,
                'curr_wrist_pitch': curr_wrist_pitch,
                'visualizer': visualizer,
                'perception_pipeline': perception_pipeline,
                'ui_manager': ui_manager,
                'tracker': tracker,
                'display_image': display_image,
            }

            stats = fsm.update(context)

            if stats.get('target_lost'):
                break

            display_image = stats.get('display_image', display_image)
            molmo_display_image = stats.get('molmo_display_image', molmo_display_image)

            # Kinematic & Fingertip UI Drawing
            if visualizer is not None and closest is not None:
                full_history_list = reconstructor.get_history_list()
                direction = 'closing'
                if len(full_history_list) > 10:
                    recent = full_history_list[-10:]
                    diff = recent[-1]['gripper']['pos_pct'] - recent[0]['gripper']['pos_pct']
                    direction = 'opening' if diff > 0 else 'closing'
                    
                predicted_fingertips = {}
                for side in ['left', 'right']:
                    pos_pred, rot_pred = visualizer.predict(side, matched_pct, direction)
                    if pos_pred is not None and rot_pred is not None:
                        predicted_fingertips[side] = {'pos': pos_pred, 'x_axis': rot_pred[:, 0], 'y_axis': rot_pred[:, 1], 'z_axis': rot_pred[:, 2]}
                        
                draw_predicted_frames(predicted_fingertips, display_image, scaled_camera_info)
                if not args.disable_suction_cups:
                    detector.aruco_to_fingertips.draw_fingertip_suction_cups(predicted_fingertips, display_image, scaled_camera_info, color=(128, 0, 0), alpha=0.4)
                
                if vis_fingertips is not None:
                    detector.aruco_to_fingertips.draw_fingertip_frames(vis_fingertips, display_image, scaled_camera_info, axis_length_in_m=0.02, draw_origins=True, write_coordinates=False)
                    if not args.disable_suction_cups:
                        detector.aruco_to_fingertips.draw_fingertip_suction_cups(vis_fingertips, display_image, scaled_camera_info, color=(255, 0, 0), alpha=0.4)

                if args.overlay_estimation_frames and active_fingertips_for_display is not None:
                    detector.aruco_to_fingertips.draw_fingertip_frames(active_fingertips_for_display, display_image, scaled_camera_info, axis_length_in_m=0.03, draw_origins=True, write_coordinates=False)
                    detector.aruco_to_fingertips.draw_fingertip_suction_cups(active_fingertips_for_display, display_image, scaled_camera_info, color=(0, 255, 0), alpha=0.6)
                    
                display_image = ui_manager.draw_fingertip_markers(display_image, fingertip_center_2d, estimation_method)

            # Display Windows
            window_name = fsm.state_instances[fsm.current_state].window_name
            
            if molmo_display_image is not None:
                # To guarantee no overlap and strictly enforce Molmo on top, Visual Servoing below,
                # we combine them into a single window.
                h1, w1 = molmo_display_image.shape[:2]
                h2, w2 = display_image.shape[:2]
                
                # Resize display_image to match molmo_display_image's width if necessary
                if w1 != w2:
                    scale = w1 / float(w2)
                    display_image_scaled = cv2.resize(display_image, (int(w2 * scale), int(h2 * scale)))
                else:
                    display_image_scaled = display_image
                    
                combined_image = np.vstack((display_image_scaled, molmo_display_image))
                ui_manager.show_window("Visual Servoing Pipeline", combined_image)
            else:
                if window_name == "Molmo Object Initialization":
                    window_name = "Camera Stream"
                ui_manager.show_window(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        fsm.send_zero_command()
        cv2.destroyAllWindows()
        print("\nStopped.")

if __name__ == '__main__':
    main()
