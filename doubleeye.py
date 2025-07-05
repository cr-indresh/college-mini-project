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

# Eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Blink parameters
blink_ratio_threshold = 0.25
click_cooldown = 0.3
double_click_cooldown = 0.5
last_click_time = 0
left_ref_height = None
right_ref_height = None

# State tracking
eyes_closed_start = 0
double_click_triggered = False

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

        # Mouse control (right eye iris tracking)
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

        # Eye detection
        left_top = landmarks[LEFT_EYE_TOP]
        left_bottom = landmarks[LEFT_EYE_BOTTOM]
        right_top = landmarks[RIGHT_EYE_TOP]
        right_bottom = landmarks[RIGHT_EYE_BOTTOM]

        # Calculate eye heights
        left_height = abs(left_top.y - left_bottom.y)
        right_height = abs(right_top.y - right_bottom.y)

        # Initialize reference heights
        if left_ref_height is None:
            left_ref_height = left_height
        if right_ref_height is None:
            right_ref_height = right_height

        # Update reference heights (only when eyes are open)
        left_ref_height = max(left_ref_height * 0.95, left_height)
        right_ref_height = max(right_ref_height * 0.95, right_height)

        # Calculate eye ratios
        left_ratio = left_height / left_ref_height
        right_ratio = right_height / right_ref_height

        # Current time and time since last click
        current_time = time.time()
        time_since_last = current_time - last_click_time

        # Detect eye states
        left_closed = left_ratio < blink_ratio_threshold
        right_closed = right_ratio < blink_ratio_threshold

        # Click handling
        if left_closed or right_closed:
            if not eyes_closed_start:
                eyes_closed_start = current_time
                
            # Double click condition (both eyes closed)
            if left_closed and right_closed:
                if not double_click_triggered and time_since_last > double_click_cooldown:
                    if current_time - eyes_closed_start > 0.2:  # Minimum hold time
                        pyautogui.doubleClick()
                        last_click_time = current_time
                        double_click_triggered = True
                        cv2.putText(frame, "DOUBLE CLICK", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Single click condition (left eye closed)
            elif left_closed and not right_closed and time_since_last > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "SINGLE CLICK", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Right click condition (right eye closed)
            elif right_closed and not left_closed and time_since_last > click_cooldown:
                pyautogui.rightClick()
                last_click_time = current_time
                cv2.putText(frame, "RIGHT CLICK", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            eyes_closed_start = 0
            double_click_triggered = False

        # Draw eye landmarks
        for landmark, color in zip([left_top, left_bottom], [(255, 0, 0), (255, 0, 0)]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, color, -1)

        for landmark, color in zip([right_top, right_bottom], [(0, 255, 0), (0, 255, 0)]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, color, -1)

    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()