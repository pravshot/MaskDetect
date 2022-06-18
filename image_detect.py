# Will read in an image and display the detections.

from imageai.Detection.Custom import CustomObjectDetection
import cv2

# Initialize the custom object detection class
# and load in the model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model_9.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()

# Read in the image
img = cv2.imread("examples/img3.jpg")

# Call the detection function
drawn_image, output_objects_array = detector.detectObjectsFromImage(input_image=img, input_type="array", output_type="array", 
                                                                          minimum_percentage_probability=50, nms_treshold=0.3)

# Show the image with detections
cv2.imshow("drawn_image", drawn_image)
cv2.waitKey(0)
cv2.imwrite('examples/img3_detections.jpg', drawn_image)
cv2.destroyAllWindows()

# Print out the detections array
print(output_objects_array)
