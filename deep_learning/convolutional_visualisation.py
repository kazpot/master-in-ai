import cv2
import matplotlib.pyplot as plt

img_path = "data/udacity_sdc.png"

bgr_img = cv2.imread(img_path)

# convert to gray scale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0, 1]
gray_img = gray_img.astype("float32") / 255

# plot image
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.show()
