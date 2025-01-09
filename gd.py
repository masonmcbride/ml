import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Gaussian function
def gaussian(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    return np.exp(-(((x - mu_x)**2 / (2 * sigma_x**2)) + ((y - mu_y)**2 / (2 * sigma_y**2))))

# Compute the gradient of the Gaussian
def gradient(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1):
    dfx = -(x - mu_x) / (sigma_x**2) * gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y)
    dfy = -(y - mu_y) / (sigma_y**2) * gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y)
    return dfx, dfy

# Create a meshgrid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = gaussian(X, Y)

# Initialize random starting point
point = np.array([np.random.uniform(-3, 3), np.random.uniform(-3, 3)])
learning_rate = 0.1

# Set up the figure and axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Initialize the scatter point
point_z = gaussian(point[0], point[1])
scatter = ax.scatter(point[0], point[1], point_z, color='red', s=100, depthshade=False)

# Update function for the animation
def update(frame):
    global point
    grad_x, grad_y = gradient(point[0], point[1])
    point += learning_rate * np.array([grad_x, grad_y])
    point_z = gaussian(point[0], point[1])
    scatter._offsets3d = ([point[0]], [point[1]], [point_z])
    return scatter,

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False)

plt.show()