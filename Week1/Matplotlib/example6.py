import numpy as np
import matplotlib.pyplot as plt

# reading image
image_path = "cat.jpeg"
image = plt.imread(image_path)

# displaying image
plt.imshow(image)
plt.show()