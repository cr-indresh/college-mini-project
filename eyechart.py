
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get screen resolution
screen_w, screen_h = pyautogui.size()

# Smoothing parameters
smooth_factor = 0.5
prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2

# Sensitivity factors for X and Y axes
sensitivity_x = 1.09  # Adjust for horizontal sensitivity
sensitivity_y = 1.05  # Adjust for vertical sensitivity

# Blink detection threshold
blink_threshold = 0.02

while True:
    # Read frame from camera
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip frame horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract eye landmarks for mouse control
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            if id == 1:  # Use a specific landmark for mouse control
                # Calculate mouse coordinates with separate X and Y sensitivity
                mouse_x = int((screen_w / frame.shape[1] * x) * sensitivity_x)
                mouse_y = int((screen_h / frame.shape[0] * y) * sensitivity_y)

                # Clamp mouse coordinates to screen bounds
                mouse_x = max(0, min(mouse_x, screen_w - 1))
                mouse_y = max(0, min(mouse_y, screen_h - 1))

                # Smooth mouse movement
                prev_mouse_x = smooth_factor * mouse_x + (1 - smooth_factor) * prev_mouse_x
                prev_mouse_y = smooth_factor * mouse_y + (1 - smooth_factor) * prev_mouse_y
                pyautogui.moveTo(prev_mouse_x, prev_mouse_y)

        # Blink detection
        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if abs(left_eye[0].y - left_eye[1].y) < blink_threshold:
            pyautogui.click()
            cv2.putText(frame, "Click Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Mouse clicked")
            pyautogui.sleep(1)  # Add delay to avoid multiple clicks

    # Display frame
    cv2.imshow("Eye Controlled Mouse", frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()