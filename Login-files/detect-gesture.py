"""
Facial Gesture Detection System
Detects and verifies facial gestures for behavioral biometric authentication
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple
import time

class GestureDetector:
    def __init__(self):
        """
        Initialize MediaPipe Face Mesh for gesture detection
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Gesture thresholds
        self.smile_threshold = 0.05
        self.eye_aspect_ratio_threshold = 0.2
        self.eyebrow_raise_threshold = 0.03
        
        # Gesture history for sequence tracking
        self.gesture_history = []
        self.max_history = 100
        
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        """
        # Vertical eye landmarks
        A = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
        B = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
        
        # Horizontal eye landmark
        C = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_smile(self, landmarks) -> bool:
        """
        Detect smile based on mouth corner positions
        """
        # Mouth corners (left and right)
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Mouth center points (top and bottom)
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        
        # Calculate mouth width vs height ratio
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        
        smile_ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        
        # Smile detected if ratio is high
        return smile_ratio > 3.5
    
    def detect_blink(self, landmarks) -> Tuple[bool, bool]:
        """
        Detect blinks in left and right eyes
        
        Returns:
            (left_eye_blink, right_eye_blink)
        """
        # Left eye landmarks
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        # Right eye landmarks  
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_indices)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_indices)
        
        left_blink = left_ear < self.eye_aspect_ratio_threshold
        right_blink = right_ear < self.eye_aspect_ratio_threshold
        
        return (left_blink, right_blink)
    
    def detect_eyebrow_raise(self, landmarks) -> bool:
        """
        Detect raised eyebrows
        """
        # Eyebrow landmarks
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[300]
        
        # Eye landmarks for reference
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate distance between eyebrow and eye
        left_distance = left_eyebrow[1] - left_eye[1]  # Y-axis distance
        right_distance = right_eyebrow[1] - right_eye[1]
        
        avg_distance = (abs(left_distance) + abs(right_distance)) / 2
        
        return avg_distance > self.eyebrow_raise_threshold
    
    def detect_head_turn(self, landmarks) -> str:
        """
        Detect head orientation (left, right, center)
        """
        # Nose tip
        nose = landmarks[1]
        
        # Face outline points
        left_face = landmarks[234]
        right_face = landmarks[454]
        
        # Calculate nose position relative to face outline
        face_width = np.linalg.norm(right_face - left_face)
        nose_offset = nose[0] - (left_face[0] + right_face[0]) / 2
        
        relative_position = nose_offset / face_width if face_width > 0 else 0
        
        if relative_position < -0.15:
            return 'left'
        elif relative_position > 0.15:
            return 'right'
        else:
            return 'center'
    
    def detect_wink(self, landmarks) -> str:
        """
        Detect winks (left, right, or none)
        """
        left_blink, right_blink = self.detect_blink(landmarks)
        
        if left_blink and not right_blink:
            return 'wink_left'
        elif right_blink and not left_blink:
            return 'wink_right'
        elif left_blink and right_blink:
            return 'blink_both'
        else:
            return 'none'
    
    def detect_all_gestures(self, frame: np.ndarray) -> Dict:
        """
        Detect all possible gestures in a frame
        
        Args:
            frame: RGB image frame
            
        Returns:
            Dictionary of detected gestures
        """
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Process frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {'face_detected': False}
        
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
        
        # Detect gestures
        gestures = {
            'face_detected': True,
            'smile': self.detect_smile(landmarks),
            'eyebrow_raise': self.detect_eyebrow_raise(landmarks),
            'head_orientation': self.detect_head_turn(landmarks),
            'wink': self.detect_wink(landmarks),
            'timestamp': time.time()
        }
        
        # Add to history
        self.gesture_history.append(gestures)
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        return gestures
    
    def verify_gesture_sequence(self, expected_sequence: List[str], 
                                tolerance: float = 1.0) -> Tuple[bool, List[str]]:
        """
        Verify if expected gesture sequence was performed
        
        Args:
            expected_sequence: List of expected gestures (e.g., ['smile', 'wink_left', 'eyebrow_raise'])
            tolerance: Time window in seconds for each gesture
            
        Returns:
            (success, performed_sequence)
        """
        if not self.gesture_history:
            return False, []
        
        performed_sequence = []
        current_gesture_idx = 0
        
        # Analyze gesture history
        for i in range(len(self.gesture_history)):
            gesture_data = self.gesture_history[i]
            
            if current_gesture_idx >= len(expected_sequence):
                break
            
            expected_gesture = expected_sequence[current_gesture_idx]
            
            # Check if expected gesture is detected
            gesture_detected = False
            
            if expected_gesture == 'smile' and gesture_data.get('smile'):
                gesture_detected = True
            elif expected_gesture == 'eyebrow_raise' and gesture_data.get('eyebrow_raise'):
                gesture_detected = True
            elif expected_gesture.startswith('wink') and gesture_data.get('wink') == expected_gesture:
                gesture_detected = True
            elif expected_gesture.startswith('head') and gesture_data.get('head_orientation') in expected_gesture:
                gesture_detected = True
            
            if gesture_detected:
                performed_sequence.append(expected_gesture)
                current_gesture_idx += 1
        
        # Check if all expected gestures were performed
        success = len(performed_sequence) == len(expected_sequence)
        
        return success, performed_sequence
    
    def create_random_challenge(self, num_gestures: int = 3) -> List[str]:
        """
        Create a random gesture challenge sequence
        
        Args:
            num_gestures: Number of gestures in sequence
            
        Returns:
            List of random gestures
        """
        available_gestures = [
            'smile',
            'eyebrow_raise',
            'wink_left',
            'wink_right',
            'head_left',
            'head_right'
        ]
        
        # Randomly select gestures
        challenge = np.random.choice(available_gestures, size=num_gestures, replace=False)
        
        return challenge.tolist()
    
    def visualize_gestures(self, frame: np.ndarray, gestures: Dict) -> np.ndarray:
        """
        Draw detected gestures on frame for visualization
        
        Args:
            frame: Input frame
            gestures: Detected gestures dictionary
            
        Returns:
            Frame with gesture annotations
        """
        annotated_frame = frame.copy()
        
        if not gestures.get('face_detected'):
            cv2.putText(annotated_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        y_offset = 30
        
        # Display detected gestures
        if gestures.get('smile'):
            cv2.putText(annotated_frame, "SMILE DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        if gestures.get('eyebrow_raise'):
            cv2.putText(annotated_frame, "EYEBROW RAISE", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        wink = gestures.get('wink')
        if wink != 'none':
            cv2.putText(annotated_frame, f"{wink.upper()}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        head_orientation = gestures.get('head_orientation')
        if head_orientation != 'center':
            cv2.putText(annotated_frame, f"HEAD: {head_orientation.upper()}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated_frame


# Example usage
if __name__ == "__main__":
    print("=== Gesture Detection System ===\n")
    
    detector = GestureDetector()
    
    print("Supported gestures:")
    print("  - Smile")
    print("  - Eyebrow raise")
    print("  - Wink (left/right)")
    print("  - Head turn (left/right)")
    print("  - Blink")
    
    print("\nExample gesture sequence:")
    example_sequence = ['smile', 'wink_left', 'eyebrow_raise']
    print(f"  {' → '.join(example_sequence)}")
    
    print("\nRandom challenge:")
    random_challenge = detector.create_random_challenge(4)
    print(f"  {' → '.join(random_challenge)}")