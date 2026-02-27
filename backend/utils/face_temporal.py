import cv2
import mediapipe as mp
import numpy as np

class FaceTemporalAnalyzer:
    """
    Tracks face landmarks across a sequence of frames to measure:
    - Eye blink frequency and naturalness.
    - Head pose stability.
    Unnatural blinking patterns or morphing anomalies increase the AI score.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # EAR (Eye Aspect Ratio) landmark indices for MediaPipe
        # Left eye: [362, 385, 387, 263, 373, 380]
        # Right eye: [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def _calculate_ear(self, landmarks, eye_indices):
        """Calculates the Eye Aspect Ratio."""
        # Horizontal distance
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        width = np.linalg.norm(p1 - p4)
        
        # Vertical distances
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        
        height1 = np.linalg.norm(p2 - p6)
        height2 = np.linalg.norm(p3 - p5)
        
        ear = (height1 + height2) / (2.0 * width + 1e-6)
        return ear

    def analyze_sequence(self, frames):
        """
        Frames: list of numpy arrays (BGR format, typically from OpenCV)
        Returns: float score (0 to 1), where 1 = Highly Unnatural/AI, 0 = Real
        """
        ear_history = []
        pose_history = []
        
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                
                # 1. Calculate EAR
                left_ear = self._calculate_ear(face.landmark, self.LEFT_EYE)
                right_ear = self._calculate_ear(face.landmark, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_history.append(avg_ear)
                
                # 2. Extract Head Pose (using nose tip, chin, eye corners, mouth corners)
                # For simplicity, we track the variance of the nose tip relative to the face size
                # to measure extreme micro-jitter or morphing instability.
                nose_tip = face.landmark[1]
                pose_history.append(np.array([nose_tip.x, nose_tip.y, nose_tip.z]))
            else:
                # Face lost tracking
                ear_history.append(None)
                pose_history.append(None)
                
        # If no faces found in the majority of frames, we can't score face temporals.
        valid_ears = [e for e in ear_history if e is not None]
        valid_poses = [p for p in pose_history if p is not None]
        
        if len(valid_ears) < len(frames) * 0.5:
            return 0.5 # Neutral fallback if face tracking fails mostly
            
        # Analysis Compute
        score = 0.0
        
        # 1. Blink Analysis (EAR variance)
        # Real videos have natural variance (blinks). AI videos often have 0 variance (staring deadpan) 
        # or extreme high-frequency jitter (glitching eyes).
        ear_variance = np.var(valid_ears)
        
        if ear_variance < 0.0001:
            # Suspiciously static (deepfake staring)
            score += 0.4
        elif ear_variance > 0.01:
            # Glitching eye morphs
            score += 0.5
            
        # 2. Pose Jitter Analysis
        # Deepfakes often have high-frequency micro-jitter in 3D face alignment.
        if len(valid_poses) > 1:
            pose_diffs = [np.linalg.norm(valid_poses[i] - valid_poses[i-1]) for i in range(1, len(valid_poses))]
            avg_jitter = np.mean(pose_diffs)
            
            # If jitter is extremely high between adjacent frames, 
            # it indicates temporal mesh instability (face morph failing)
            if avg_jitter > 0.05:
                score += 0.6
                
        return min(1.0, score)
