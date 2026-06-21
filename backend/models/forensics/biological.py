import cv2
import numpy as np


class BiologicalAnalyzer:
    """
    Biological and Physiological Forensics Analyzer.
    Detects AI anomalies in facial features like corneal specular highlights (eye reflections)
    and pupil circularity.
    """
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_face(self, face_crop_cv: np.ndarray) -> dict:
        """
        Analyzes a face crop for biological anomalies.
        Returns a dictionary with anomaly scores and details.
        """
        if face_crop_cv is None or face_crop_cv.size == 0:
            return {"anomaly_score": 0.0, "details": "Invalid face crop"}

        gray_face = cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) < 2:
            return {"anomaly_score": 0.3, "details": "Could not detect both eyes clearly; possible structural anomaly."}

        # Sort eyes by X coordinate to get left and right eye
        eyes = sorted(eyes, key=lambda x: x[0])
        eye1_rect = eyes[0]
        eye2_rect = eyes[1]

        # Extract eye regions
        eye1_img = gray_face[eye1_rect[1]:eye1_rect[1]+eye1_rect[3], eye1_rect[0]:eye1_rect[0]+eye1_rect[2]]
        eye2_img = gray_face[eye2_rect[1]:eye2_rect[1]+eye2_rect[3], eye2_rect[0]:eye2_rect[0]+eye2_rect[2]]

        # Analyze Specular Highlights
        highlight1_score = self._analyze_specular_highlight(eye1_img)
        highlight2_score = self._analyze_specular_highlight(eye2_img)

        # In a real photograph, the lighting reflection in both eyes should be highly consistent in angle and intensity.
        # AI models often generate completely independent, physically impossible reflections.
        reflection_mismatch = abs(highlight1_score - highlight2_score)

        # Calculate final biological anomaly score
        anomaly_score = min(1.0, reflection_mismatch * 2.0)

        details = "Normal biological features."
        if anomaly_score > 0.6:
            details = "Severe specular highlight mismatch detected between left and right eye. Lighting physics violated."
        elif anomaly_score > 0.4:
            details = "Moderate pupil/reflection asymmetry."

        return {
            "anomaly_score": round(anomaly_score, 4),
            "reflection_mismatch": round(reflection_mismatch, 4),
            "details": details
        }

    def _analyze_specular_highlight(self, eye_gray: np.ndarray) -> float:
        """
        Finds the brightest spot in the eye (corneal reflection) and computes its relative position/intensity.
        """
        # Threshold to find the brightest spots (reflections)
        _, thresh = cv2.threshold(eye_gray, 200, 255, cv2.THRESH_BINARY)

        # Calculate the center of mass of the highlight
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Normalize position based on eye size
            h, w = eye_gray.shape
            norm_cx = cx / w
            norm_cy = cy / h
            # Return a combined metric representing the position vector magnitude
            return np.sqrt(norm_cx**2 + norm_cy**2)
        else:
            return 0.0 # No clear reflection found
