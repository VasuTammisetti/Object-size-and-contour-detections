# Object-size-and-contour-detections
Fundamental before proceeding with the installation of the libraries that allow the use of Aruco Marker. In this case, you have to make sure you have installed Contrib OpenCV, attention is not enough just installing OpenCV, it has to be specifically with Contrib otherwise you may have lost file errors or something similar.

If you have not installed it, run this command from the terminal, it does not matter which operating system you are using.

Identify the object in space



Before proceeding with the measurement, we need to find the object in space and obtain its coordinates. For this project, it is not a fundamental part of understanding the topic so I have already prepared a file called object_detector.py that you can download and import.

from object_detector import *

# Load Object Detector
detector = HomogeneousBgDetector()

...

contours = detector.detect_objects(img)

...

We use a for loop to extract the arrays with all points delimiting the identified objects. Not having perfect geometric shapes we have to reduce the delimitation of the object to a simple rectangle, as in this case.




# Draw objects boundaries
for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    
    
    
    
    For details in coading please follow the source code file uploaded in OpenCV 
    
    
    Sample results 
    
    
    
