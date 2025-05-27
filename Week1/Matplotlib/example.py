import numpy as np
import matplotlib.pyplot as plt

# Generate 100 points between 0 and 10
x = np.linspace(0, 10, 100)

# Calculate sin(x)
y = np.sin(x)

# Plot the curve
plt.plot(x, y)

# Label the axes
plt.xlabel("x")
plt.ylabel("sin(x)")

# Title
plt.title("Sin Wave")

# Show plot
plt.show()
