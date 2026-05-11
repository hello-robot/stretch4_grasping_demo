# Stretch 4 Grasping Demo

This repository contains a grasping demo for the Stretch 4 mobile manipulator from Hello Robot. Grasping is based on RGB and depth images from Stretch 4's gripper camera. 

Perception uses the [Molmo 2](https://github.com/allenai/molmo2) Vision-Language Model (VLM) to output pixel coordinates for a target object described with text. The pixel coordinates are then used to prompt the [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) to segment the target object, after which SAM 2 tracks and segments the target object over time. 

A finite state machine (FSM) consisting of a sequence of visual servoing behaviors controls the robot's motions. The behaviors use the segmentation mask output by SAM 2, the depth image from the wrist-mounted camera, and estimates of the gripper's fingertip frames of reference to decide how to move the robot.  

The FSM behaviors use three control modes provided by [flying gripper control](https://github.com/hello-robot/stretch4_flying_gripper_control/):

* Mode 1: Gripper Frame Relative Motions
* Mode 2: Gripper Frame Projected into the Base Frame Relative Motions
* Mode 3: Direct Joint-Space Control of the Joints (Relative and Absolute Motions)

More details about the FSM and tunable parameters for the behaviors can be found in [visual_servo_fsm_params.py](visual_servo_fsm_params.py).

## Prerequisites

To run this code on a remote desktop computer, you must have the following local repositories cloned in the same parent directory (e.g., `~/repos/`):

1.  **`stretch4_gripper_modeling_and_control`**: Provides the core gripper kinematic models, telemetry utilities, and command scripts.
2.  **`stretch4_flying_gripper_control`**: (If applicable) Provides the underlying control interfaces for the end-of-arm tooling.
> [!NOTE]
> **Architecture Note:** Both Molmo 2 and SAM 2.1 are loaded dynamically on the fly using the Hugging Face `transformers` library. You do *not* need to clone their respective repositories or manually download checkpoints.

## Installation on Remote Desktop (Ubuntu 24.04 + RTX 5090)

The visual servoing perception system used by `grasping_demo.py` and `recv_and_molmo_sam2_gripper_images.py` requires a powerful GPU. These instructions are tailored for an NVIDIA GeForce RTX 5090 running on Ubuntu 24.04.

### 1. Create a Virtual Environment

Navigate to the `stretch4_grasping_demo` directory and create a new Python virtual environment:

```bash
cd ~/repos/stretch4_grasping_demo
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch (CUDA 12.8)

The RTX 5090 (Blackwell architecture) requires at least CUDA 12.8. You must install a PyTorch nightly build to support this architecture natively. Ensure your `.venv` is activated, then run:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support
pip install --pre --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

*(Note: Once a stable release of PyTorch with built-in cu128 or higher support becomes available, you can replace this with the standard stable installation command.)*

### 3. Install Standard Dependencies

Install the remaining required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Install Local Repositories

Install the related Stretch 4 packages in editable mode:

```bash
pip install -e ../stretch4_gripper_modeling_and_control/
```

```bash
pip install -e ../stretch4_flying_gripper_control/
```


## Running the Code

### Networking Configuration

This grasping demo relies on a high-bandwidth, low-latency connection between the robot and the remote computer. A dedicated, high-performance WiFi access point is recommended.

After you have ensured that the robot and the remote computer are connected and able to communicate with each other, you can proceed to edit the IP addresses in `gripper_networking.py` to match the robot's IP address and the remote computer's IP address. 

This file should be edited on both the robot and the remote computer. The file is in the `src/stretch4_gripper_modeling_and_control/` directory of the stretch4_gripper_modeling_and_control repository.

You can see the file on GitHub via the following link: 

[stretch4_gripper_modeling_and_control/gripper_networking.py](https://github.com/hello-robot/stretch4_gripper_modeling_and_control/blob/main/src/stretch4_gripper_modeling_and_control/gripper_networking.py)

The two variable to change are at the top of the file, as shown in the following excerpt:

```bash
# Set these values for your network
robot_ip = '100.90.83.97'
remote_computer_ip = '100.69.89.24'
```

Once networking is configured, you can proceed to copy the gripper calibration files from the robot to the remote computer.

### Sync the Gripper Calibration Files

On your remote computer, run the following script to copy the gripper calibration files from the robot to the desktop:

```bash
python3 ../stretch4_gripper_modeling_and_control/sync_calibration_models.py
```

This tool automatically connects to the robot using the IP specified in `gripper_networking.py`, downloads the calibration files in a single `scp` command (minimizing password prompts), and extracts the correct `HELLO_FLEET_ID` directly from the downloaded data. It then automatically organizes the models into your desktop's local `~/stretch_user/<robot_id>/calibration_gripper` directory, keeping both machines in sync without requiring you to manually specify the robot's ID.


### Start Robot-Side Services

On the Stretch robot (in the `stretch4_gripper_modeling_and_control` repository directory), open two separate terminals and run:

**Terminal 1 (Receive Commands):**
```bash
python3 recv_and_execute_gripper_commands.py --remote
```

**Terminal 2 (Publish Images & States):**
```bash
python3 send_gripper_images_and_joint_states.py --remote
```

### Test the Perception System (Desktop)

Prior to making the robot move via closed-loop control, you should first test the perception system. On your remote desktop computer, ensure your `.venv` is activated, then run the following script with an OBJECT_DESCRIPTION that describes the object you want the robot to grasp. Prior to running the script, the target object should be in view of the robot's gripper camera. This script will prompt Molmo 2, use the pixel coordinates it provides to prompt SAM 2, and then use SAM 2 to track and segment the target object over time. The results will be visualized in a window.

While the script is running, you can move the object around, occlude it, deform it, and otherwise manipulate it to test the robustness of the perception system.

> A text description of the target object (OBJECT_DESCRIPTION) should be provided on the command line.

```bash
source ~/repos/stretch4_grasping_demo/.venv/bin/activate
python3 recv_and_molmo_sam2_gripper_images.py --remote OBJECT_DESCRIPTION
```

The object description text is used to prompt the Molmo 2 VLM. Examples of object descriptions that have been used successfully follow.

**Example Object Descriptions:**
- "sunscreen"
- "cleaning wipe container"
- "coffee mug"
- "white plastic cup"
- "black rubber rocket with red nozzle"
- "brown paper cup"

**The VLM Prompt**

The full Molmo 2 prompt is defined by the `get_molmo_pointing_prompt(object_description)` function in [vlm_utils.py](vlm_utils.py). Advanced users can edit this prompt to better match their application.

### Run Visual Servoing (Desktop)

If the previous test of the perception system was successful at tracking and segmenting the target object at a high frame rate with low latency, you can proceed to run the visual servoing finite state machine (FSM) that commands the robot to move and attempts to grasp the object. 

First, make sure that the perception system test code is no longer running.

Then, in a terminal on your remote desktop, activate the environment and run the visual servoing FSM. As with the perception test, a text description of the target object (OBJECT_DESCRIPTION) should be provided on the command line. 

> [WARNING] This visual servoing demo does not avoid obstacles. It is strictly driven by joint states and depth and RGB camera images from the wrist-mounted camera. Make sure that the scene is clear of obstacles before running this script. You should also be prepared to stop the code and run-stop the robot via the button on the robot's head.  

```bash
source ~/repos/stretch4_grasping_demo/.venv/bin/activate
python3 grasping_demo.py --remote OBJECT_DESCRIPTION
```

**Optional Arguments:**
- `--move_to_front`: By default, the robot will approach the object directly. Passing this flag enables an exploration phase where the robot moves sideways to find the front face of the object (minimizing the apparent width) before approaching.


### Testing Manual Gripper Control (Optional)

To test the gripper control manually, you can plug a gamepad dongle into the desktop and run the `send_gripper_commands.py` script from the installed model repository:

```bash
cd ../stretch4_gripper_modeling_and_control
python3 send_gripper_commands.py --remote
```
