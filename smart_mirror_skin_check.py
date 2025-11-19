import cv2
import numpy as np

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def optimize_camera(cap):
    """Set better camera parameters"""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

def analyze_skin(face_roi):
    """Enhanced skin analysis with multiple metrics"""
    # Convert to HSV for better skin analysis
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]  # Value channel (brightness)
    
    # Apply adaptive thresholding
    blur = cv2.GaussianBlur(v_channel, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Calculate affected area
    affected_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    affected_ratio = affected_pixels / total_pixels
    percentage = round(affected_ratio * 100, 1)
    
    # Texture analysis
    laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
    
    # Determine condition
    if affected_ratio > 0.15:
        return f"⚠ Severe dryness/acne ({percentage}%)", (0, 0, 255), thresh
    elif affected_ratio > 0.10:
        return f"⚠ Moderate issues ({percentage}%)", (0, 165, 255), thresh
    elif affected_ratio > 0.07:
        return f"⚠ Mild irritation ({percentage}%)", (0, 255, 255), thresh
    else:
        return f"✅ Healthy skin ({percentage}%)", (0, 255, 0), thresh

# Initialize camera with optimized settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
optimize_camera(cap)

cv2.namedWindow("Skin Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Skin Analysis", 1000, 700)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)  # Enhance details
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(200, 200))
    
    message = "No face detected"
    color = (255, 255, 255)
    analysis_display = np.zeros((200, 300, 3), dtype=np.uint8)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        message, color, thresh = analyze_skin(face_roi)
        
        # Create analysis visualization
        analysis_display = cv2.resize(thresh, (300, 200))
        analysis_display = cv2.cvtColor(analysis_display, cv2.COLOR_GRAY2BGR)
        cv2.putText(analysis_display, "Problem Areas", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display main frame with results
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Press ESC to exit", (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Resize analysis_display to match frame height before concatenation
    if analysis_display is not None:
        analysis_display_resized = cv2.resize(analysis_display, (300, frame.shape[0]))
        combined = np.hstack((frame, analysis_display_resized))
    else:
        combined = frame
    
    cv2.imshow("Skin Analysis", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()