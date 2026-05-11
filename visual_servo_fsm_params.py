"""
Parameters for the Visual Servo Finite State Machine

======================================================================
Visual Servo Finite State Machine Sequence
======================================================================
  [INITIALIZE] 
       | (Wait for gripper to fully open)
       v
  [FIND_OBJECT] 
       | (VLM identifies target pixel)
       v
  [LOOK_AT_OBJECT] 
       | (Wait for tracking to settle smoothly)
       v
  [MOVE_TO_OBJECT_HEIGHT] 
       | (Translate lift using feedback to align gripper finger plane with object)
       v
  [MOVE_TO_FRONT_OF_OBJECT] 
       | *Note: Skipped by default. Use --move_to_front to enable.*
       | (Move sideways to minimize the object's width, finding the front face)
       | (If the object is elongated, this can find the narrow front of the object)
       | (Currently, a cylindrical object can result in this state continuing indefinitely)
       v
  [APPROACH_OBJECT] 
       | (Translate base forward until target distance is reached)
       v
  [LOOK_AT_OBJECT] 
       | (Wait for tracking to settle smoothly)
       v
  [ALIGN_GRASP] 
       | (Rotate wrist roll to match object orientation)
       v
  [PRE_GRASP] 
       | (Open gripper slightly wider than the object)
       v
  [APPROACH_GRASP] 
       | (Move forward until fingertips align with target depth)
       v
  [EXECUTE_GRASP] 
       | (Close gripper until stall is detected)
       v
  [LIFT_OBJECT] 
       | (Translate lift upwards to raise the object)
       v
  [RETRACT_GRASP] 
       | (Translate base backward to clear the grasping area)
       v
  [END SEQUENCE]
======================================================================
"""


# ==========================================
# GENERAL SERVOING PARAMETERS
# ==========================================

# Factor to dampen rotational speed during precise alignment states (e.g. LOOK_AT_OBJECT, ALIGN_GRASP)
PRECISE_ALIGNMENT_ROT_DAMPING = 0.4

# Additional damping multiplier for the yaw joint specifically during ALIGN_GRASP to prevent single-axis instability
ALIGN_GRASP_YAW_DAMPING_MULTIPLIER = 0.2

# Proportional gain for visual servoing yaw alignment (centering object horizontally)
KP_YAW = 1.5

# Proportional gain for visual servoing pitch alignment (centering object vertically)
KP_PITCH = 1.5

# Proportional gain for visual servoing roll alignment (aligning gripper to object grasp axis)
KP_ROLL = 1.5

# ==========================================
# INITIALIZE
# ==========================================
# Overview: Opens the gripper to a designated width to prepare for grasping.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Gripper pos_pct must be >= INITIALIZE_TARGET_POS_PCT.

# Target percentage for gripper aperture opening (100% is fully open)
INITIALIZE_TARGET_POS_PCT = 90.0

# Speed command for opening the gripper
INITIALIZE_SPEED = 100.0

# Acceleration command for opening the gripper
INITIALIZE_ACCEL = 100.0

# ==========================================
# FIND_OBJECT
# ==========================================
# Overview: Prompts the VLM to find the target object's 2D coordinate.
# Control Mode: N/A (Handled in main script)
# Transition Criteria: Target coordinate found successfully.

# (No FSM-specific tuning parameters)

# ==========================================
# MOVE_TO_OBJECT_HEIGHT
# ==========================================
# Overview: Translates lift to align height, zeroes wrist pitch and roll, and centers object.
# Control Mode: 2 (Projected Base Frame Relative IK)
# Transition Criteria: Pitch, roll, yaw, and height errors within tolerances.

# Proportional gain for vertical height adjustment
KP_HEIGHT = 4.0

# Maximum vertical speed allowed for the lift (m/s)
MAX_HEIGHT_SPEED_M_S = 0.5

# Tolerance for vertical height error to consider settled (meters)
HEIGHT_TOLERANCE_M = 0.02

# Tolerance for wrist pitch angle to consider settled (radians)
PITCH_TOLERANCE_RAD = 0.1

# Tolerance for wrist roll angle to consider settled (radians)
ROLL_TOLERANCE_RAD = 0.1

# Minimum number of consecutive frames the height error must be within tolerance before transitioning
HEIGHT_SETTLE_FRAMES = 3

# Minimum assumed depth of the object to prevent division by zero when computing normalized yaw error (meters)
MIN_DEPTH_FOR_YAW_M = 0.05

# ==========================================
# MOVE_TO_FRONT_OF_OBJECT
# ==========================================
# Overview: Move sideways to minimize the object's width (find the front face).
# Control Mode: 2 (Projected Base Frame Relative IK)
# Transition Criteria: Width stops decreasing and starts increasing, or settle time reached.

# Speed at which to move sideways while exploring the object width (m/s)
FRONT_EXPLORE_SPEED_M_S = 0.5

# Tolerance to determine if the width is increasing during the initial probe phase (meters)
FRONT_WIDTH_PROBE_TOLERANCE_M = 0.01

# Tolerance to determine if the width has increased past the minimum found width (meters)
FRONT_WIDTH_TOLERANCE_M = 0.01

# Minimum number of consecutive frames the base must be stopped before transitioning
FRONT_SETTLE_FRAMES = 3

# Minimum number of frames to probe in a direction before determining if it's the wrong direction
FRONT_MIN_PROBE_FRAMES = 3

# Minimum number of frames to move continuously before checking if we've passed the minimum width
FRONT_MIN_MOVE_FRAMES = 5

# ==========================================
# PRE_APPROACH_LOOK
# ==========================================
# Overview: Pauses to allow the camera's visual servoing to center the object.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Time-based settling.
# Note: This state is currently unused in the main sequence flow.

# Factor to dampen rotational speed specifically during the PRE_APPROACH_LOOK state
PRE_APPROACH_LOOK_ROT_DAMPING = 0.5

# ==========================================
# APPROACH_OBJECT
# ==========================================
# Overview: Moves the gripper forward toward the object.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Euclidean distance between gripper and object <= target dist.

# Forward speed to approach the object (m/s)
APPROACH_OBJECT_FORWARD_SPEED = 0.8

# The target Euclidean distance between the gripper and the object to stop the approach (meters)
APPROACH_OBJECT_TARGET_DIST_M = 0.30

# ==========================================
# LOOK_AT_OBJECT
# ==========================================
# Overview: Pauses to allow the visual tracking to lock onto the object firmly.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Time-based settling.

# Tolerance for rotational velocity commands to consider the tracking settled
LOOK_AT_ROT_VEL_TOLERANCE = 0.1

# Minimum number of consecutive frames tracking must be settled before transitioning
LOOK_AT_SETTLE_FRAMES = 4

# ==========================================
# ALIGN_GRASP
# ==========================================
# Overview: Rotates the wrist roll to align the fingertips with the object's grasp axis.
# Control Mode: 1 (Gripper Frame Relative IK) / Open-Loop FF
# Transition Criteria: Roll error < tolerance for settle time.

# Tolerance for roll alignment error to consider settled (radians)
ALIGN_ANGLE_TOLERANCE_RAD = 0.087

# Threshold for ratio of mask bounding box axes to consider an object circular
CIRCULAR_MASK_AXIS_RATIO_THRESHOLD = 0.85

# Minimum number of consecutive frames the alignment must be within tolerance before transitioning
ALIGN_SETTLE_FRAMES = 5

# ==========================================
# PRE_GRASP
# ==========================================
# Overview: Adjusts the gripper aperture to comfortably surround the object.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Gripper width is within tolerance of target width for settle time.

# Extra margin added to the object's estimated width to set the target pre-grasp aperture (meters)
PRE_GRASP_WIDTH_MARGIN_M = 0.03

# Fast feedforward speed for initial gripper adjustment
PRE_GRASP_FF_SPEED = 100.0

# Fast feedforward acceleration for initial gripper adjustment
PRE_GRASP_FF_ACCEL = 100.0

# Speed command for small, ongoing gripper adjustments
PRE_GRASP_SPEED = 10.0

# Acceleration command for small, ongoing gripper adjustments
PRE_GRASP_ACCEL = 10.0

# Maximum position percentage displacement allowed per frame for small adjustments
PRE_GRASP_MAX_DISP = 5.0

# Tolerance percentage for aperture position error to consider settled
PRE_GRASP_TOLERANCE_PCT = 15.0

# Minimum number of consecutive frames the aperture must be within tolerance before transitioning
PRE_GRASP_SETTLE_FRAMES = 5

# Deadband percentage where no further small aperture adjustments are made
APERTURE_FEEDBACK_DEADBAND_PCT = 2.0

# ==========================================
# APPROACH_GRASP
# ==========================================
# Overview: Moves the base forward slowly to place the fingertips around the object.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Fingertip Z-depths pass the object's Z-depth + margin.

# Depth margin past the object's estimated contact point to target for the fingertips (meters)
APPROACH_GRASP_DEPTH_MARGIN_M = 0.0

# Forward speed used when the gripper is far from the final grasp depth (m/s)
APPROACH_GRASP_FAST_SPEED = 0.3

# Forward speed used when the gripper is very close to the final grasp depth (m/s)
APPROACH_GRASP_SLOW_SPEED = 0.1

# Distance threshold from the final depth at which to switch from fast to slow speed (meters)
APPROACH_GRASP_SLOW_DIST_M = 0.04

# ==========================================
# EXECUTE_GRASP
# ==========================================
# Overview: Closes the gripper to grab the object.
# Control Mode: 1 (Gripper Frame Relative IK)
# Transition Criteria: Gripper position stalls (indicates successful grasp) or misses.

# Target finger deflection (visual aperture - kinematic aperture) to consider the grasp secure (meters)
EXECUTE_GRASP_TARGET_DEFLECTION_M = 0.02

# Speed command for closing the gripper
EXECUTE_GRASP_SPEED = 50.0

# Acceleration command for closing the gripper
EXECUTE_GRASP_ACCEL = 50.0

# Maximum position percentage displacement allowed per frame for the grasp
EXECUTE_GRASP_MAX_DISP = 10.0

# Deadband percentage for grasp position error
EXECUTE_GRASP_DEADBAND = 2.0

# Minimum valid grasp width below which the gripper defaults to fully closed (meters)
MIN_VALID_GRASP_WIDTH_M = 0.05

# Default fully closed position target percentage if the target width is too small
DEFAULT_CLOSE_POS_PCT = -300.0

# Minimum percentage change in position required to not consider the gripper stalled
EXEC_STALL_DIFF_PCT = 1.0

# Maximum time the gripper can remain stalled before transitioning (seconds)
EXEC_STALL_TIME_S = 0.5

# ==========================================
# LIFT_OBJECT
# ==========================================
# Overview: Lifts the arm up to raise the grasped object off the surface.
# Control Mode: 3 (Joint-Space Direct Control)
# Transition Criteria: Lift joint has traveled the specified distance.

# Vertical distance to lift the object (meters)
LIFT_OBJECT_DIST_M = 0.2

# Maximum allowed absolute height for the lift joint (meters)
MAX_LIFT_HEIGHT_M = 1.05

# Tolerance for the lift height reaching its target before transitioning (meters)
LIFT_HEIGHT_TOLERANCE_M = 0.01

# ==========================================
# RETRACT_GRASP
# ==========================================
# Overview: Moves the base backward to clear the grasping area.
# Control Mode: 2 (Projected Base Frame Relative IK)
# Transition Criteria: Time-based settling (estimated from desired distance).
# Note: Mode 2 utilizes weighted damped pseudoinverse velocity control, which 
# prevents the use of open-loop move_by commands. Distance is used to compute time.

# Backward speed for retracting the base (m/s)
RETRACT_BACK_SPEED_M_S = 1.0

# Time duration to retract the base (seconds)
RETRACT_TIME_S = 3.0
