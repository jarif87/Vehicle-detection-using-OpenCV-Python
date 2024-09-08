import cv2
import numpy as np
from time import sleep

# Set minimum width and height for vehicle detection
min_width = 80  
min_height = 80  

offset = 6  # Reduced offset to be more sensitive to line crossing
line_position = 550  # Position of the detection line

delay = 60  # Time delay between frames
detections = []  # List to store the detected object centers

vehicle_count = 0  # Counter for vehicles
counted_vehicles = []  # List to store already counted vehicles (avoid double counting)

# Function to calculate the center of a bounding box
def get_center(x, y, w, h):
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    return center_x, center_y

# Open the video file
cap = cv2.VideoCapture("video.mp4")
background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()  # Read video frame
    if not ret:
        break  # Exit loop if no frame is captured
    
    # Set a delay between frames
    time_delay = float(1 / delay)  
    sleep(time_delay)
    
    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    
    # Apply background subtraction
    img_subtracted = background_subtractor.apply(blur)
    
    # Perform dilation and morphological closing to fill gaps
    dilated = cv2.dilate(img_subtracted, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # Find contours from the processed frame
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the detection line on the color frame
    cv2.line(frame, (25, line_position), (1200, line_position), (255, 127, 0), 3)
    
    # List to store new detections
    current_detections = []

    for contour in contours:
        # Get the bounding box for each detected contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Validate the contour size
        if w >= min_width and h >= min_height:
            # Draw rectangle around detected vehicles on the original color frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get the center point of the detected vehicle
            center = get_center(x, y, w, h)
            current_detections.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            
            # Check if the vehicle center crosses the detection line
            if line_position - offset < center[1] < line_position + offset:
                if center not in counted_vehicles:  # Check if the vehicle is already counted
                    vehicle_count += 1
                    counted_vehicles.append(center)  # Add vehicle center to counted vehicles list
                    cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)  # Change line color
                    print(f"Vehicle detected: {vehicle_count}")
    
    # Display the vehicle count on the screen
    cv2.putText(frame, f"VEHICLE COUNT: {vehicle_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    # Show the original color video with bounding boxes
    cv2.imshow("Vehicle detection using OpenCV Python", frame)
    
    # Exit on pressing the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources and close windows
cv2.destroyAllWindows()
cap.release()
