
# Python program to explain cv2.imshow() method  
  
# importing cv2  
import cv2
import time
  
# path  
path = 'fork.jpg'
  
# Reading an image in default mode 
image = cv2.imread(path) 
print(image)  
# Window name in which image is displayed 
window_name = 'image'
image = cv2.resize(image,(100,100))
# Using cv2.imshow() method  
# Displaying the image  
cv2.imshow(window_name, image) 
  
#waits for user to press any key  
#(this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()  
