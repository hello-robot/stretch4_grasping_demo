import argparse
import os

def add_shared_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('object_description', type=str, help='Text description of the object to track')
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.')
    parser.add_argument('--sam2_model_id', type=str, default='facebook/sam2.1-hiera-large', help='Hugging Face model ID for SAM 2.1')
    parser.add_argument('--tracking_mode', type=str, choices=['image', 'video', 'camera'], default='video', help='SAM 2 tracking implementation.')
    parser.add_argument('--model', type=str, default=None, help='Path to the model planar YAML file. Activates kinematic visualization.')
    parser.add_argument('--disable_suction_cups', action='store_true', help='Disable rendering solid topological visualization of the 3D suction cups.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose printouts.')

def resolve_model_path(args, robot_id):
    """
    Resolves the model path if none is provided.
    Checks ~/stretch_user/<ROBOT_ID>/calibration_gripper/ directory for latest_model_planar.yaml first,
    then falls back to the fleet calibration default.
    Raises FileNotFoundError if the kinematic model or corresponding mirror transforms are missing.
    """
    if getattr(args, 'model', None) is None:
        if robot_id:
            user_path = os.path.expanduser(f"~/stretch_user/{robot_id}/calibration_gripper/latest_model_planar.yaml")
            if os.path.exists(user_path):
                args.model = user_path
                print(f"No model path provided. Defaulting to user calibration for {robot_id}: {args.model}")

    if getattr(args, 'model', None) is None:
        try:
            from stretch4_gripper_modeling_and_control import calibration_utils as cu
            args.model = cu.get_default_model_path()
            if args.model:
                print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
        except ImportError:
            print("Warning: Could not import calibration_utils for model path resolution.")
            
    if getattr(args, 'model', None) is None or not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Failed to locate the required kinematic model planar YAML file. "
            f"Please ensure the default model exists in ~/stretch_user/{robot_id}/calibration_gripper/ "
            f"or provide a valid path via the --model argument."
        )

    expected_robot_id = robot_id
    if expected_robot_id:
        import yaml
        try:
            with open(args.model, 'r') as f:
                model_data = yaml.safe_load(f)
                if model_data is not None and 'metadata' in model_data:
                    model_robot_id = model_data.get('metadata', {}).get('robot_id')
                    if model_robot_id and expected_robot_id != model_robot_id:
                        raise ValueError(
                            f"Robot ID mismatch! The requested robot ID is '{expected_robot_id}', "
                            f"but the kinematic model at {args.model} was fit for '{model_robot_id}'. "
                            "Please verify your calibration files."
                        )
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"Warning: Could not verify robot_id in {args.model}: {e}")

        
    mirror_path_1 = os.path.join(os.path.dirname(args.model), f"mirror_transforms_{os.path.basename(args.model)}")
    mirror_path_2 = os.path.join(os.path.dirname(args.model), "latest_mirror_transforms.yaml")
    
    if os.path.exists(mirror_path_1):
        args.mirror_path = mirror_path_1
    elif os.path.exists(mirror_path_2):
        args.mirror_path = mirror_path_2
    else:
        raise FileNotFoundError(
            f"Failed to locate the required mirror transforms file. Checked for {mirror_path_1} and {mirror_path_2}. "
            "This file is required alongside the kinematic model planar YAML file."
        )
    
    return args
