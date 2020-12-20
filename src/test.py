from loss import loss
from model import get_vgg_model
from train import get_img, parse_image
import cv2

img = get_img("images/content/chicago.jpg")
img = parse_image(img)
print(img.shape)
cv2.imwrite("test.png",cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR))