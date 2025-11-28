"""
Demo script for testing InsightFace with detailed facial feature analysis.

This script detects all faces in an image and extracts detailed facial features including:
- Bounding boxes
- 106 facial landmarks (detailed facial features)
- Eye Aspect Ratio (EAR) - for blink detection
- Mouth Aspect Ratio (MAR) - for mouth openness detection
- Eyebrow position analysis
- Jaw position analysis
- Detection confidence scores

Usage:
    python test_insightface.py <image_path>

Example:
    python test_insightface.py test_image.jpg
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


# Landmark indices for 106-point model (approximate mapping)
# Note: The exact indices may vary based on the model version
# These are common patterns observed in 106-point models

# Eye landmarks (approximate)
LEFT_EYE_INDICES = list(range(35, 42))  # 7 points for left eye
RIGHT_EYE_INDICES = list(range(42, 49))  # 7 points for right eye

# Mouth landmarks (approximate)
MOUTH_OUTER_INDICES = list(range(52, 72))  # 20 points for outer mouth

# Eyebrow landmarks (approximate)
LEFT_EYEBROW_INDICES = list(range(13, 21))  # 8 points for left eyebrow
RIGHT_EYEBROW_INDICES = list(range(21, 29))  # 8 points for right eyebrow

# Jaw landmarks (approximate)
JAW_INDICES = list(range(0, 13))  # 13 points along jawline


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1-p6 are the eye landmark points.
    Lower EAR indicates closed/closing eyes (blinking).
    Typical threshold: EAR < 0.25 indicates a blink.

    Args:
        eye_landmarks: Array of eye landmark coordinates (at least 6 points)

    Returns:
        Eye aspect ratio value
    """
    if len(eye_landmarks) < 6:
        return 0.0

    # Vertical eye distances
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])

    # Horizontal eye distance
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth openness detection.

    MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p0-p6||)

    Higher MAR indicates mouth is open (talking, yawning).
    Typical threshold: MAR > 0.6 indicates mouth is open.

    Args:
        mouth_landmarks: Array of mouth landmark coordinates (at least 12 points)

    Returns:
        Mouth aspect ratio value
    """
    if len(mouth_landmarks) < 12:
        return 0.0

    # Vertical mouth distances
    A = euclidean_distance(mouth_landmarks[2], mouth_landmarks[10])
    B = euclidean_distance(mouth_landmarks[4], mouth_landmarks[8])

    # Horizontal mouth distance
    C = euclidean_distance(mouth_landmarks[0], mouth_landmarks[6])

    if C == 0:
        return 0.0

    mar = (A + B) / (2.0 * C)
    return mar


def analyze_eyebrow_position(eyebrow_landmarks: np.ndarray, eye_landmarks: np.ndarray) -> dict:
    """
    Analyze eyebrow position relative to eye.

    Args:
        eyebrow_landmarks: Array of eyebrow landmark coordinates
        eye_landmarks: Array of eye landmark coordinates

    Returns:
        Dictionary with eyebrow metrics
    """
    if len(eyebrow_landmarks) == 0 or len(eye_landmarks) == 0:
        return {"average_height": 0.0, "raised": False}

    # Calculate average y-coordinate of eyebrow and eye
    eyebrow_center_y = np.mean(eyebrow_landmarks[:, 1])
    eye_center_y = np.mean(eye_landmarks[:, 1])

    # Distance between eyebrow and eye (negative means eyebrow is above eye)
    distance = eyebrow_center_y - eye_center_y

    # Consider eyebrow "raised" if distance is relatively large
    # (this is a simple heuristic, threshold may need adjustment)
    raised = distance < -15  # Eyebrow is significantly above eye

    return {
        "average_height": -distance,  # Positive value = eyebrow above eye
        "raised": raised
    }


def analyze_jaw_position(jaw_landmarks: np.ndarray) -> dict:
    """
    Analyze jaw position and angle.

    Args:
        jaw_landmarks: Array of jaw landmark coordinates

    Returns:
        Dictionary with jaw metrics
    """
    if len(jaw_landmarks) < 3:
        return {"angle": 0.0, "width": 0.0}

    # Calculate jaw width (distance between leftmost and rightmost points)
    leftmost = jaw_landmarks[0]
    rightmost = jaw_landmarks[-1]
    jaw_width = euclidean_distance(leftmost, rightmost)

    # Calculate jaw angle (angle of jawline)
    # Using first 3 and last 3 points to estimate angle
    if len(jaw_landmarks) >= 6:
        left_vector = jaw_landmarks[2] - jaw_landmarks[0]
        right_vector = jaw_landmarks[-1] - jaw_landmarks[-3]

        # Calculate angles relative to horizontal
        left_angle = np.degrees(np.arctan2(left_vector[1], left_vector[0]))
        right_angle = np.degrees(np.arctan2(right_vector[1], right_vector[0]))

        avg_angle = (abs(left_angle) + abs(right_angle)) / 2.0
    else:
        avg_angle = 0.0

    return {
        "angle": avg_angle,
        "width": jaw_width
    }


def detect_faces_detailed(image_path: str, output_path: str = None):
    """
    Detect all faces in an image using InsightFace with detailed 106 landmarks.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated image (optional)

    Returns:
        List of detected faces with their attributes
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("Error: insightface not installed.")
        print("Install it using: uv add insightface")
        print("You may also need: uv add onnxruntime or uv add onnxruntime-gpu")
        sys.exit(1)

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    print(f"Image size: {img.shape[1]}x{img.shape[0]} (width x height)")

    # Initialize FaceAnalysis with 106 landmark detection
    print("\nInitializing InsightFace with 106-landmark detection...")
    print("This may take a moment on first run (downloading models)...")

    try:
        # Use detection + 106 landmark model
        app = FaceAnalysis(
            allowed_modules=['detection', 'landmark_2d_106'],
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        print(f"\nWarning: Could not load 106-landmark model: {e}")
        print("Falling back to standard 5-point detection...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))

    # Detect faces
    print("\nDetecting faces...")
    faces = app.get(img)

    print(f"\n{'='*70}")
    print(f"Found {len(faces)} face(s) in the image")
    print(f"{'='*70}\n")

    # Create a copy for visualization
    img_annotated = img.copy()

    # Process each detected face
    for idx, face in enumerate(faces, 1):
        print(f"Face #{idx}:")
        print(f"  Detection confidence: {face.det_score:.4f}")

        # Bounding box
        bbox = face.bbox.astype(int)
        print(f"  Bounding box: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
        print(f"  Face size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} pixels")

        # Draw bounding box
        cv2.rectangle(img_annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     (0, 255, 0), 2)

        # Add face number label
        label_y = bbox[1] - 10 if bbox[1] > 20 else bbox[1] + 20
        cv2.putText(img_annotated, f"Face {idx}", (bbox[0], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check if 106 landmarks are available
        has_106_landmarks = hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None

        if has_106_landmarks:
            landmarks = face.landmark_2d_106.astype(int)
            print(f"\n  Detailed Facial Analysis (106 landmarks):")
            print(f"  Total landmarks detected: {len(landmarks)}")

            # Draw all landmarks (small dots)
            for point in landmarks:
                cv2.circle(img_annotated, tuple(point), 1, (200, 160, 75), -1)

            # Analyze eyes
            print(f"\n  Eye Analysis:")
            try:
                left_eye_points = landmarks[LEFT_EYE_INDICES] if len(landmarks) > max(LEFT_EYE_INDICES) else None
                right_eye_points = landmarks[RIGHT_EYE_INDICES] if len(landmarks) > max(RIGHT_EYE_INDICES) else None

                if left_eye_points is not None:
                    left_ear = calculate_eye_aspect_ratio(left_eye_points)
                    print(f"    Left Eye Aspect Ratio (EAR): {left_ear:.3f}")
                    print(f"    Left Eye Status: {'CLOSED/BLINKING' if left_ear < 0.25 else 'OPEN'}")

                    # Draw left eye landmarks
                    for point in left_eye_points:
                        cv2.circle(img_annotated, tuple(point), 2, (255, 0, 0), -1)

                if right_eye_points is not None:
                    right_ear = calculate_eye_aspect_ratio(right_eye_points)
                    print(f"    Right Eye Aspect Ratio (EAR): {right_ear:.3f}")
                    print(f"    Right Eye Status: {'CLOSED/BLINKING' if right_ear < 0.25 else 'OPEN'}")

                    # Draw right eye landmarks
                    for point in right_eye_points:
                        cv2.circle(img_annotated, tuple(point), 2, (255, 0, 0), -1)

                # Overall blink detection
                if left_eye_points is not None and right_eye_points is not None:
                    avg_ear = (left_ear + right_ear) / 2.0
                    print(f"    Average EAR: {avg_ear:.3f}")
                    if avg_ear < 0.25:
                        print(f"    >>> BLINK DETECTED <<<")
                        cv2.putText(img_annotated, "BLINKING", (bbox[0], bbox[3] + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                print(f"    Could not analyze eyes: {e}")

            # Analyze mouth
            print(f"\n  Mouth Analysis:")
            try:
                mouth_points = landmarks[MOUTH_OUTER_INDICES] if len(landmarks) > max(MOUTH_OUTER_INDICES) else None

                if mouth_points is not None:
                    mar = calculate_mouth_aspect_ratio(mouth_points)
                    print(f"    Mouth Aspect Ratio (MAR): {mar:.3f}")
                    print(f"    Mouth Status: {'OPEN' if mar > 0.6 else 'CLOSED'}")

                    if mar > 0.6:
                        print(f"    >>> MOUTH OPEN (talking/yawning) <<<")
                        cv2.putText(img_annotated, "MOUTH OPEN", (bbox[0], bbox[3] + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    # Draw mouth landmarks
                    for point in mouth_points:
                        cv2.circle(img_annotated, tuple(point), 2, (0, 0, 255), -1)
            except Exception as e:
                print(f"    Could not analyze mouth: {e}")

            # Analyze eyebrows
            print(f"\n  Eyebrow Analysis:")
            try:
                left_eyebrow = landmarks[LEFT_EYEBROW_INDICES] if len(landmarks) > max(LEFT_EYEBROW_INDICES) else None
                right_eyebrow = landmarks[RIGHT_EYEBROW_INDICES] if len(landmarks) > max(RIGHT_EYEBROW_INDICES) else None

                if left_eyebrow is not None and left_eye_points is not None:
                    left_brow_analysis = analyze_eyebrow_position(left_eyebrow, left_eye_points)
                    print(f"    Left Eyebrow Height: {left_brow_analysis['average_height']:.1f} pixels above eye")
                    print(f"    Left Eyebrow Status: {'RAISED' if left_brow_analysis['raised'] else 'NORMAL'}")

                    # Draw left eyebrow
                    for point in left_eyebrow:
                        cv2.circle(img_annotated, tuple(point), 2, (0, 255, 255), -1)

                if right_eyebrow is not None and right_eye_points is not None:
                    right_brow_analysis = analyze_eyebrow_position(right_eyebrow, right_eye_points)
                    print(f"    Right Eyebrow Height: {right_brow_analysis['average_height']:.1f} pixels above eye")
                    print(f"    Right Eyebrow Status: {'RAISED' if right_brow_analysis['raised'] else 'NORMAL'}")

                    # Draw right eyebrow
                    for point in right_eyebrow:
                        cv2.circle(img_annotated, tuple(point), 2, (0, 255, 255), -1)
            except Exception as e:
                print(f"    Could not analyze eyebrows: {e}")

            # Analyze jaw
            print(f"\n  Jaw Analysis:")
            try:
                jaw_points = landmarks[JAW_INDICES] if len(landmarks) > max(JAW_INDICES) else None

                if jaw_points is not None:
                    jaw_analysis = analyze_jaw_position(jaw_points)
                    print(f"    Jaw Width: {jaw_analysis['width']:.1f} pixels")
                    print(f"    Jaw Angle: {jaw_analysis['angle']:.1f} degrees")

                    # Draw jaw landmarks
                    for point in jaw_points:
                        cv2.circle(img_annotated, tuple(point), 2, (255, 0, 255), -1)
            except Exception as e:
                print(f"    Could not analyze jaw: {e}")

        else:
            # Fallback to 5-point landmarks
            print(f"\n  Using basic 5-point landmarks (106-landmark model not available)")
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps.astype(int)
                landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']

                for lm_idx, (point, name) in enumerate(zip(landmarks, landmark_names)):
                    print(f"    {name}: ({point[0]}, {point[1]})")

                    colors = [
                        (255, 0, 0),    # Left eye - Blue
                        (255, 0, 0),    # Right eye - Blue
                        (0, 255, 255),  # Nose - Yellow
                        (0, 0, 255),    # Left mouth - Red
                        (0, 0, 255)     # Right mouth - Red
                    ]
                    cv2.circle(img_annotated, (point[0], point[1]), 3, colors[lm_idx], -1)

        print()

    # Save annotated image
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / (input_path.stem + "_detailed.jpg"))

    cv2.imwrite(str(output_path), img_annotated)
    print(f"Annotated image saved to: {output_path}")

    # Display legend
    print("\nVisualization Legend:")
    print("  - Green rectangles: Face bounding boxes")
    print("  - Blue dots: Eye landmarks")
    print("  - Red dots: Mouth landmarks")
    print("  - Yellow dots: Eyebrow landmarks")
    print("  - Magenta dots: Jaw landmarks")
    print("  - Small beige dots: All other facial landmarks")

    return faces


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_insightface.py <image_path>")
        print("\nExample:")
        print("  python test_insightface.py test_image.jpg")
        print("\nThis script uses InsightFace's 106-landmark model to detect:")
        print("  - Eye blinks (via Eye Aspect Ratio)")
        print("  - Mouth openness (via Mouth Aspect Ratio)")
        print("  - Eyebrow position (raised or normal)")
        print("  - Jaw position and angle")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Run face detection with detailed analysis
    faces = detect_faces_detailed(image_path, output_path)

    print(f"\n{'='*70}")
    print(f"Summary: Successfully analyzed {len(faces)} face(s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
