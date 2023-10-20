# import cv2


# img = cv2.imread('../sketchify/photo.jpeg')
# gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# invert  = cv2.bitwise_not(gr)
# blur = cv2.GaussianBlur(invert,(21,21),0)
# invertBlur = cv2.bitwise_not(blur)
# sketch = cv2.divide(gr,invertBlur,scale=256.0)
# cv2.imwrite('../sketchify/sketch.jpg', sketch)

import cv2
import requests
import numpy as np
from io import BytesIO

# URL of the image you want to process
image_url = 'https://imgs.search.brave.com/AH6-v9gVjCwOA4JUkdWGW34dQAyoe2KLO8Ex6_30tpE/rs:fit:860:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy9k/L2QyL1lhc2hfYXRf/dGhlXyVFMiU4MCU5/OEtHRiVFMiU4MCU5/OV9QcmVzc19NZWV0/X0luX0NoZW5uYWlf/KGNyb3BwZWQpLmpw/Zw'

# Download the image from the URL
response = requests.get(image_url)
image_bytes = BytesIO(response.content)
img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(gr)
blur = cv2.GaussianBlur(invert, (21, 21), 0)
invertBlur = cv2.bitwise_not(blur)
sketch = cv2.divide(gr, invertBlur, scale=256.0)

# Save the processed image to a file
cv2.imwrite('../sketchify/sketch.jpg', sketch)
