# Will use a camera(webcam) to detect objects in real time.

from imageai.Detection.Custom import CustomObjectDetection
import cv2

# Initialize the custom object detection class
# and load in the model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model_9.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()

# Captures frames from camera
cap = cv2.VideoCapture(0)

while True:
    # Reads frame from camera
    ret, img = cap.read()
    # Calls detection function
    drawn_image, output_objects_array = detector.detectObjectsFromImage(input_image=img, input_type="array", output_type="array", 
                                                                          minimum_percentage_probability=50, nms_treshold=0.3)
    # Display the image in a window
    cv2.imshow('detections', drawn_image)
    
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()