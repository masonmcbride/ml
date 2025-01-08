import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, solve_triangular
import time

# Generate input data
np.random.seed(42)  # Set seed for reproducibility
X = np.random.rand(100, 1)  # 100 samples, 1 feature for 2D plot
true_beta = np.random.rand(1, 1)  # True coefficient
y = X @ true_beta + 0.1 * np.random.randn(100, 1)  # Add some noise

# Start timing
start_time = time.time()

# Step 1: QR decomposition of X
Q, R = qr(X, mode='economic')

# Step 2: Compute Q^T y
QTy = np.dot(Q.T, y)

# Step 3: Solve R * beta = Q^T y
beta = solve_triangular(R, QTy)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label="Data Points", alpha=0.7)
plt.plot(X, X @ beta, color='red', label="Regression Line", linewidth=2)
plt.title("Linear Regression: Data and Fitted Line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid()
plt.show()

# Display timing result
print(f"Elapsed time: {elapsed_time:.5f} seconds")
