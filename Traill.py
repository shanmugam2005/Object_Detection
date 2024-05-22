import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect changes between frames
def detect_changes(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference between frames
    diff = cv2.absdiff(prev_gray, current_gray)
    
    # Apply threshold to highlight significant changes
    
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Count changed pixels
    changed_pixels = np.count_nonzero(threshold)
    
    return changed_pixels

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Read the first frame
ret, prev_frame = cap.read()

# Initialize lists to store time and changed pixel counts
time_list = []
changed_pixels_list = []

while True:
    # Read a frame from the camera
    ret, current_frame = cap.read()
    
    if not ret:
        break
    
    # Detect changes between frames
    changed_pixels = detect_changes(prev_frame, current_frame)
    
    # Store time and changed pixel count
    time_list.append(len(time_list) + 1)
    changed_pixels_list.append(changed_pixels)
    
    # Update the previous frame
    prev_frame = current_frame.copy()
    
    # Display the frame
    cv2.imshow('Frame', current_frame)
    
    # Plot the changes
    plt.plot(time_list, changed_pixels_list, color='b')
    plt.xlabel('Time')
    plt.ylabel('Changed Pixels')
    plt.title('Changes Over Time')
    plt.draw()
    plt.pause(0.01)
    
    # Clear the plot for the next iteration
    plt.clf()
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print(changed_pixels_list )
# Release the capture
cap.release()
cv2.destroyAllWindows()
