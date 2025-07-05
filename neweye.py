import cv2
import mediapipe as mp
import pyautogui
import time

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

# Sensitivity factors
sensitivity_x = 1.09
sensitivity_y = 1.05

# Left eye blink parameters
LEFT_EYE_TOP = 159    # Left eye upper lid landmark
LEFT_EYE_BOTTOM = 145  # Left eye lower lid landmark
blink_ratio_threshold = 0.25
reference_eye_height = None
click_cooldown = 0.5
last_click_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Mouse control (right eye iris tracking - unchanged)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            if id == 1:
                mouse_x = int((screen_w / frame.shape[1] * x) * sensitivity_x)
                mouse_y = int((screen_h / frame.shape[0] * y) * sensitivity_y)

                mouse_x = max(0, min(mouse_x, screen_w - 1))
                mouse_y = max(0, min(mouse_y, screen_h - 1))

                prev_mouse_x = smooth_factor * mouse_x + (1 - smooth_factor) * prev_mouse_x
                prev_mouse_y = smooth_factor * mouse_y + (1 - smooth_factor) * prev_mouse_y
                pyautogui.moveTo(prev_mouse_x, prev_mouse_y)

        # Left eye blink detection
        left_top = landmarks[LEFT_EYE_TOP]
        left_bottom = landmarks[LEFT_EYE_BOTTOM]
        
        # Calculate current eye height for left eye
        current_height = abs(left_top.y - left_bottom.y)
        
        # Initialize or update reference height
        if reference_eye_height is None:
            reference_eye_height = current_height
        else:
            # Only update reference when left eye is open
            reference_eye_height = max(reference_eye_height * 0.95, current_height)

        # Calculate eye closure ratio
        eye_ratio = current_height / reference_eye_height

        # Detect blink only in left eye
        current_time = time.time()
        if eye_ratio < blink_ratio_threshold:
            if current_time - last_click_time > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "LEFT EYE CLICK", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw left eye landmarks
        for landmark in [left_top, left_bottom]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue for left eye

    cv2.imshow("Left Eye Controlled Click", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()