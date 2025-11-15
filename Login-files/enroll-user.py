"""
User Enrollment Script
Process training photos and enroll users in the facial authentication system
"""

import os
import cv2
import argparse
from pathlib import Path
from facial_auth_system import FacialEmbeddingSystem
from gesture_detector import GestureDetector
import json

class UserEnrollment:
    def __init__(self, training_data_dir: str):
        """
        Initialize enrollment system
        
        Args:
            training_data_dir: Directory containing user training photos
                              Expected structure: training_data/user_name/image1.jpg, image2.jpg, ...
        """
        self.training_data_dir = Path(training_data_dir)
        self.auth_system = FacialEmbeddingSystem()
        self.gesture_detector = GestureDetector()
        
        # Create tables (if they don't exist)
        self.auth_system.create_aws_tables()
        
    def scan_training_data(self) -> dict:
        """
        Scan training data directory and organize by user
        
        Returns:
            Dictionary mapping user_id to list of image paths
        """
        users_data = {}
        
        if not self.training_data_dir.exists():
            print(f"✗ Training data directory not found: {self.training_data_dir}")
            return users_data
        
        # Iterate through user directories
        for user_dir in self.training_data_dir.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                image_paths = []
                
                # Get all image files for this user
                for img_file in user_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_paths.append(str(img_file))
                
                if image_paths:
                    users_data[user_id] = image_paths
                    print(f"Found {len(image_paths)} images for user: {user_id}")
        
        return users_data
    
    def enroll_all_users(self, users_data: dict, gesture_patterns: dict = None):
        """
        Enroll all users from training data
        
        Args:
            users_data: Dictionary mapping user_id to image paths
            gesture_patterns: Optional dictionary mapping user_id to gesture sequences
        """
        total_users = len(users_data)
        successful_enrollments = 0
        
        print(f"\n=== Starting Enrollment for {total_users} Users ===\n")
        
        for user_id, image_paths in users_data.items():
            print(f"\nEnrolling: {user_id}")
            print(f"  Images: {len(image_paths)}")
            
            # Get gesture pattern for this user if available
            gesture_seq = gesture_patterns.get(user_id) if gesture_patterns else None
            
            # Enroll user
            success = self.auth_system.enroll_user(
                user_id=user_id,
                image_paths=image_paths,
                gesture_sequence=gesture_seq
            )
            
            if success:
                successful_enrollments += 1
                print(f"  ✓ Enrollment successful")
                if gesture_seq:
                    print(f"  ✓ Gesture pattern: {' → '.join(gesture_seq)}")
            else:
                print(f"  ✗ Enrollment failed")
        
        print(f"\n=== Enrollment Complete ===")
        print(f"Success: {successful_enrollments}/{total_users} users")
        
    def create_gesture_patterns_interactive(self, users_data: dict) -> dict:
        """
        Interactively create gesture patterns for each user
        
        Args:
            users_data: Dictionary of user IDs
            
        Returns:
            Dictionary mapping user_id to gesture sequences
        """
        gesture_patterns = {}
        
        available_gestures = [
            'smile',
            'eyebrow_raise', 
            'wink_left',
            'wink_right',
            'head_left',
            'head_right'
        ]
        
        print("\n=== Gesture Pattern Setup ===")
        print("Available gestures:", ', '.join(available_gestures))
        print()
        
        for user_id in users_data.keys():
            print(f"\nCreate gesture pattern for {user_id}")
            print("Options:")
            print("  1. Use random pattern (recommended)")
            print("  2. Custom pattern")
            print("  3. Skip (no gesture authentication)")
            
            choice = input("Select option (1-3): ").strip()
            
            if choice == '1':
                pattern = self.gesture_detector.create_random_challenge(num_gestures=3)
                gesture_patterns[user_id] = pattern
                print(f"✓ Random pattern: {' → '.join(pattern)}")
                
            elif choice == '2':
                print(f"Enter gestures separated by commas (e.g., smile,wink_left,eyebrow_raise):")
                custom_input = input("> ").strip()
                pattern = [g.strip() for g in custom_input.split(',')]
                
                # Validate gestures
                if all(g in available_gestures for g in pattern):
                    gesture_patterns[user_id] = pattern
                    print(f"✓ Custom pattern: {' → '.join(pattern)}")
                else:
                    print("✗ Invalid gestures. Skipping gesture auth for this user.")
            
            else:
                print("Skipping gesture authentication")
        
        return gesture_patterns
    
    def test_enrollment(self, test_image_path: str) -> dict:
        """
        Test authentication with a single image
        
        Args:
            test_image_path: Path to test image
            
        Returns:
            Authentication result
        """
        print(f"\n=== Testing Authentication ===")
        print(f"Test image: {test_image_path}")
        
        result = self.auth_system.authenticate_user(test_image_path)
        
        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  User ID: {result['user_id']}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
        print(f"  Message: {result['message']}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        
        return result
    
    def export_enrollment_summary(self, output_file: str = 'enrollment_summary.json'):
        """
        Export enrollment summary to JSON file
        """
        # Scan for enrolled users (mock for now)
        summary = {
            'enrollment_date': str(self.auth_system),
            'total_users': 0,
            'model_version': self.auth_system.model_name,
            'detector': self.auth_system.detector_backend
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Enrollment summary exported to {output_file}")


def main():
    """
    Main enrollment workflow
    """
    parser = argparse.ArgumentParser(description='Enroll users in facial authentication system')
    parser.add_argument('--training-dir', type=str, default='./training_data',
                       help='Directory containing training photos')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Test image path for authentication verification')
    parser.add_argument('--auto-gestures', action='store_true',
                       help='Automatically generate random gesture patterns')
    
    args = parser.parse_args()
    
    # Initialize enrollment
    enrollment = UserEnrollment(args.training_dir)
    
    # Scan training data
    users_data = enrollment.scan_training_data()
    
    if not users_data:
        print("No training data found! Please organize photos in:")
        print("  training_data/")
        print("    ├── user1/")
        print("    │   ├── photo1.jpg")
        print("    │   ├── photo2.jpg")
        print("    ├── user2/")
        print("    │   ├── photo1.jpg")
        return
    
    # Create gesture patterns
    if args.auto_gestures:
        gesture_patterns = {}
        for user_id in users_data.keys():
            gesture_patterns[user_id] = enrollment.gesture_detector.create_random_challenge(3)
    else:
        gesture_patterns = enrollment.create_gesture_patterns_interactive(users_data)
    
    # Enroll all users
    enrollment.enroll_all_users(users_data, gesture_patterns)
    
    # Test if test image provided
    if args.test_image and os.path.exists(args.test_image):
        enrollment.test_enrollment(args.test_image)
    
    # Export summary
    enrollment.export_enrollment_summary()
    
    print("\n✓ Enrollment process complete!")


if __name__ == "__main__":
    main()