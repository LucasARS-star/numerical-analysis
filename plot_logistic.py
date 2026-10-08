import numpy as np
import matplotlib.pyplot as plt

def logistic_model(x, L=1, k=1, x0=0):
    """
    Logistic model function.
    L: the curve's maximum value
    k: the logistic growth rate or steepness of the curve
    x0: the x value of the sigmoid's midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))

# Generate x values
x_values = np.linspace(-10, 10, 400)

# Calculate y values using the standard logistic function
y_values = logistic_model(x_values)

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Logistic Curve (L=1, k=1, x0=0)', color='blue', linewidth=2)
plt.title('Logistic Model')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the plot
plt.savefig('logistic_curve.png')
print("Graph saved as logistic_curve.png")
