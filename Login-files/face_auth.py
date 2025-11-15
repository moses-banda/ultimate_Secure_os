"""
Facial Embedding Authentication System
Generates secure facial embeddings and stores them in AWS cloud
"""

import cv2
import numpy as np
from deepface import DeepFace
import boto3
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

class FacialEmbeddingSystem:
    def __init__(self, aws_region='us-east-1'):
        """
        Initialize the facial embedding system with AWS integration
        """
        # AWS DynamoDB setup
        self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Table names
        self.embeddings_table_name = 'user_face_embeddings'
        self.gestures_table_name = 'user_gesture_patterns'
        
        # DeepFace model configuration
        self.model_name = 'Facenet512'  # 512-dimensional embeddings
        self.detector_backend = 'retinaface'  # Best for accuracy
        
        # Security parameters
        self.similarity_threshold = 0.6  # Cosine similarity threshold
        self.liveness_check_enabled = True
        
    def create_aws_tables(self):
        """
        Create DynamoDB tables for storing embeddings and gesture patterns
        """
        try:
            # Embeddings table
            embeddings_table = self.dynamodb.create_table(
                TableName=self.embeddings_table_name,
                KeySchema=[
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            # Gesture patterns table
            gestures_table = self.dynamodb.create_table(
                TableName=self.gestures_table_name,
                KeySchema=[
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            print("✓ AWS tables created successfully")
            return True
        except Exception as e:
            print(f"Tables might already exist or error: {e}")
            return False
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate facial embedding from image
        
        Args:
            image_path: Path to face image
            
        Returns:
            512-dimensional embedding vector or None if failed
        """
        try:
            # Extract embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                print(f"✓ Generated embedding with shape: {embedding.shape}")
                return embedding
            else:
                print("✗ No face detected in image")
                return None
                
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            return None
    
    def generate_multi_angle_embeddings(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings from multiple angles of the same person
        
        Args:
            image_paths: List of image paths (different angles)
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for img_path in image_paths:
            embedding = self.generate_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
        
        print(f"✓ Generated {len(embeddings)} embeddings from {len(image_paths)} images")
        return embeddings
    
    def enroll_user(self, user_id: str, image_paths: List[str], 
                    gesture_sequence: List[str] = None) -> bool:
        """
        Enroll a new user in the system
        
        Args:
            user_id: Unique user identifier
            image_paths: List of face images (multiple angles)
            gesture_sequence: Optional gesture password sequence
            
        Returns:
            Success status
        """
        try:
            # Generate embeddings from all images
            embeddings = self.generate_multi_angle_embeddings(image_paths)
            
            if not embeddings:
                print("✗ No valid embeddings generated")
                return False
            
            # Calculate average embedding (template)
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Store embeddings in DynamoDB
            table = self.dynamodb.Table(self.embeddings_table_name)
            
            # Convert numpy array to list for JSON storage
            embedding_data = {
                'user_id': user_id,
                'embedding_template': avg_embedding.tolist(),
                'individual_embeddings': [emb.tolist() for emb in embeddings],
                'enrollment_date': datetime.now().isoformat(),
                'model_version': self.model_name,
                'num_samples': len(embeddings)
            }
            
            table.put_item(Item=embedding_data)
            
            # Store gesture sequence if provided
            if gesture_sequence:
                self.store_gesture_pattern(user_id, gesture_sequence)
            
            print(f"✓ User {user_id} enrolled successfully with {len(embeddings)} samples")
            return True
            
        except Exception as e:
            print(f"✗ Error enrolling user: {e}")
            return False
    
    def store_gesture_pattern(self, user_id: str, gesture_sequence: List[str]):
        """
        Store user's gesture-based password
        
        Args:
            user_id: User identifier
            gesture_sequence: List of gestures (e.g., ['smile', 'wink_left', 'raise_eyebrows'])
        """
        try:
            table = self.dynamodb.Table(self.gestures_table_name)
            
            gesture_data = {
                'user_id': user_id,
                'gesture_sequence': gesture_sequence,
                'sequence_hash': hashlib.sha256(
                    ''.join(gesture_sequence).encode()
                ).hexdigest(),
                'created_date': datetime.now().isoformat()
            }
            
            table.put_item(Item=gesture_data)
            print(f"✓ Gesture pattern stored for user {user_id}")
            
        except Exception as e:
            print(f"✗ Error storing gesture pattern: {e}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def authenticate_user(self, image_path: str, gesture_sequence: List[str] = None) -> Dict:
        """
        Authenticate a user attempting to login
        
        Args:
            image_path: Path to login attempt image
            gesture_sequence: Performed gesture sequence
            
        Returns:
            Authentication result with user_id if successful
        """
        start_time = time.time()
        
        try:
            # Generate embedding from login image
            login_embedding = self.generate_embedding(image_path)
            
            if login_embedding is None:
                return {
                    'success': False,
                    'message': 'No face detected',
                    'processing_time': time.time() - start_time
                }
            
            # Retrieve all user embeddings from DynamoDB
            table = self.dynamodb.Table(self.embeddings_table_name)
            response = table.scan()
            
            best_match = None
            best_similarity = 0.0
            
            # Compare against all enrolled users
            for item in response['Items']:
                template_embedding = np.array(item['embedding_template'])
                similarity = self.calculate_similarity(login_embedding, template_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = item
            
            # Check if similarity meets threshold
            if best_similarity >= self.similarity_threshold:
                user_id = best_match['user_id']
                
                # Verify gesture sequence if provided
                if gesture_sequence:
                    gesture_valid = self.verify_gesture_sequence(user_id, gesture_sequence)
                    if not gesture_valid:
                        return {
                            'success': False,
                            'message': 'Gesture sequence verification failed',
                            'processing_time': time.time() - start_time
                        }
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'similarity_score': best_similarity,
                    'message': 'Authentication successful',
                    'processing_time': time.time() - start_time
                }
            else:
                return {
                    'success': False,
                    'message': 'No matching user found',
                    'best_similarity': best_similarity,
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Authentication error: {str(e)}',
                'processing_time': time.time() - start_time
            }
    
    def verify_gesture_sequence(self, user_id: str, performed_sequence: List[str]) -> bool:
        """
        Verify that performed gesture sequence matches stored pattern
        
        Args:
            user_id: User identifier
            performed_sequence: Gestures performed during login
            
        Returns:
            True if sequence matches
        """
        try:
            table = self.dynamodb.Table(self.gestures_table_name)
            response = table.get_item(Key={'user_id': user_id})
            
            if 'Item' in response:
                stored_sequence = response['Item']['gesture_sequence']
                return stored_sequence == performed_sequence
            
            return False
            
        except Exception as e:
            print(f"✗ Error verifying gesture: {e}")
            return False
    
    def liveness_detection(self, video_frames: List[np.ndarray]) -> bool:
        """
        Perform liveness detection on video frames
        
        Args:
            video_frames: List of consecutive frames from camera
            
        Returns:
            True if real person detected
        """
        # Placeholder for liveness detection
        # In production, implement:
        # 1. Texture analysis (real skin vs screen)
        # 2. 3D depth analysis
        # 3. Micro-expression detection
        # 4. Eye tracking
        
        if len(video_frames) < 3:
            return False
        
        # Simple motion detection as basic liveness check
        motion_detected = False
        for i in range(1, len(video_frames)):
            diff = cv2.absdiff(video_frames[i-1], video_frames[i])
            if np.mean(diff) > 5:  # Threshold for motion
                motion_detected = True
                break
        
        return motion_detected


# Example usage and testing
if __name__ == "__main__":
    print("=== Facial Embedding Authentication System ===\n")
    
    # Initialize system
    auth_system = FacialEmbeddingSystem()
    
    print("System initialized with:")
    print(f"  - Model: {auth_system.model_name}")
    print(f"  - Detector: {auth_system.detector_backend}")
    print(f"  - Similarity threshold: {auth_system.similarity_threshold}")
    print("\nReady for user enrollment and authentication!")
    
    # Note: Actual AWS credentials needed for cloud operations
    # For local testing, mock the DynamoDB operations