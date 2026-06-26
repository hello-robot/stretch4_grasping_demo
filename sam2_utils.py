import os
os.environ["TQDM_DISABLE"] = "1"
import cv2
import numpy as np
import torch

def load_sam2_model(model_id, tracking_mode):
    print(f"Loading SAM 2.1 from Hugging Face ({model_id}) using {tracking_mode} mode...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if tracking_mode == 'image':
        from transformers import Sam2Model, Sam2Processor
        try:
            sam2_model = Sam2Model.from_pretrained(model_id).to(device)
            sam2_processor = Sam2Processor.from_pretrained(model_id)
            return sam2_model, sam2_processor
        except Exception as e:
            print(f"Failed to load SAM 2 Image Model: {e}")
            return None, None
            
    elif tracking_mode in ['video', 'camera']:
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        try:
            sam2_model = Sam2VideoModel.from_pretrained(model_id).to(device)
            sam2_processor = Sam2VideoProcessor.from_pretrained(model_id)
            return sam2_model, sam2_processor
        except Exception as e:
            print(f"Failed to load SAM 2 Video Model: {e}")
            print("Falling back to image mode...")
            return load_sam2_model(model_id, 'image')
            
    return None, None

class SAM2Tracker:
    def __init__(self, predictor, tracking_mode, model=None):
        # We overload the arguments so we can pass (model, processor) or (processor)
        # Note: In visually_servo_gripper.py, it expects `tracker = SAM2Tracker(predictor, args.tracking_mode)`
        # We will assume `predictor` is a tuple of (model, processor) here.
        if isinstance(predictor, tuple) and len(predictor) == 2:
            self.model = predictor[0]
            self.processor = predictor[1]
        else:
            self.model = model
            self.processor = predictor
            
        self.tracking_mode = tracking_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.last_mask = None
        
        # specific to video mode
        self.video_inference_session = None
        self.video_frame_idx = 0
        
    def initialize_tracking(self, rgb_image, point):
        input_points = [[[[float(point[0]), float(point[1])]]]]
        input_labels = [[[1]]]
        
        if self.tracking_mode == 'image':
            inputs = self.processor(images=rgb_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            self.last_mask = (outputs.pred_masks[0, 0, 0] > 0.0).cpu().numpy().squeeze()
            
        elif self.tracking_mode in ['video', 'camera']:
            self.video_frame_idx = 0
            self.video_inference_session = self.processor.init_video_session(inference_device=self.device)
            
            inputs = self.processor(images=[rgb_image], return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"][0]
            
            self.video_inference_session.add_new_frame(pixel_values, frame_idx=self.video_frame_idx)
            
            # Add points to the session
            self.processor.process_new_points_or_boxes_for_video_frame(
                self.video_inference_session,
                frame_idx=self.video_frame_idx,
                obj_ids=[1],
                input_points=input_points,
                input_labels=input_labels,
                original_size=rgb_image.shape[:2]
            )
            
            # Propagate to get the mask
            with torch.no_grad():
                generator = self.model.propagate_in_video_iterator(self.video_inference_session, start_frame_idx=self.video_frame_idx, max_frame_num_to_track=1)
                for output in generator:
                    masks = self.processor.post_process_masks(
                        [output.pred_masks], 
                        original_sizes=[rgb_image.shape[:2]]
                    )
                    self.last_mask = (masks[0][0] > 0.0).cpu().numpy().squeeze()
                    break
        
        return self.last_mask

    def update_tracking(self, rgb_image):
        """
        Updates the tracker with a new frame.
        Returns: last_mask, target_lost (boolean)
        """
        target_lost = False
        H, W = rgb_image.shape[:2]
        
        if self.tracking_mode == 'image':
            y, x = np.where(self.last_mask > 0)
            
            if len(y) < 10:
                target_lost = True
            else:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                margin = 15
                x_min, x_max = max(0, x_min - margin), min(W, x_max + margin)
                y_min, y_max = max(0, y_min - margin), min(H, y_max + margin)
                
                input_boxes = [[[[float(x_min), float(y_min), float(x_max), float(y_max)]]]]
                
                inputs = self.processor(images=rgb_image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                self.last_mask = (outputs.pred_masks[0, 0, 0] > 0.0).cpu().numpy().squeeze()
                
                if np.sum(self.last_mask) < 20:
                    target_lost = True
                    
        elif self.tracking_mode in ['video', 'camera']:
            self.video_frame_idx += 1
            
            inputs = self.processor(images=[rgb_image], return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"][0]
            
            self.video_inference_session.add_new_frame(pixel_values, frame_idx=self.video_frame_idx)
            
            with torch.no_grad():
                generator = self.model.propagate_in_video_iterator(self.video_inference_session, start_frame_idx=self.video_frame_idx, max_frame_num_to_track=1)
                for output in generator:
                    masks = self.processor.post_process_masks(
                        [output.pred_masks], 
                        original_sizes=[rgb_image.shape[:2]]
                    )
                    self.last_mask = (masks[0][0] > 0.0).cpu().numpy().squeeze()
                    break
                generator.close()  # Force generator cleanup
                    
            if np.sum(self.last_mask) < 20:
                target_lost = True

            # Prune video inference session history to prevent memory leak
            session = self.video_inference_session
            history_limit = 32
            
            # Keep keys in processed_frames to preserve session.num_frames / dictionary length,
            # but replace old heavy frame tensors with None to free GPU memory.
            if session.processed_frames:
                for k in list(session.processed_frames.keys()):
                    if k < self.video_frame_idx:
                        session.processed_frames[k] = None
                
            # Keep only the last 16 frames of outputs/histories
            for obj_idx in list(session.output_dict_per_obj.keys()):
                non_cond = session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"]
                keys_to_delete = [k for k in non_cond.keys() if k < self.video_frame_idx - history_limit]
                for k in keys_to_delete:
                    non_cond.pop(k, None)
                    
                tracked = session.frames_tracked_per_obj[obj_idx]
                keys_to_delete = [k for k in tracked.keys() if k < self.video_frame_idx - history_limit]
                for k in keys_to_delete:
                    tracked.pop(k, None)

        return self.last_mask, target_lost
