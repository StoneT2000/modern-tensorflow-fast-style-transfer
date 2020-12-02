import transform
import matplotlib.pyplot as plt
model = transform.transform_net()
print(model.summary())

img = plt.imread('./chicago.jpg')
plt.imshow(img)