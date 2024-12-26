import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic equation: y = ax^2 + bx + c
a, b, c = 1, -3, 2  # Example coefficients
x = np.linspace(-10, 10, 400)  # x values
y = a * x**2 + b * x + c  # Calculate y values

# Plot the quadratic equation
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f"y = {a}x² + {b}x + {c}")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # x-axis
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # y-axis
plt.title("Graph of the Quadratic Equation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.5)
plt.legend()
plt.show()
