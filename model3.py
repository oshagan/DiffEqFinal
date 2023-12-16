import numpy as np
import matplotlib.pyplot as plt

# Function to define the system of differential equations
def equations(t, A, p, y, beta, theta, g, y0):
    dAdt = g * p * (y * p**(1/beta))
    dpdt = theta * (y - y0) * p
    dydt = (g * p - beta * theta * (y - y0)) * y
    return dAdt, dpdt, dydt

# Euler's method for solving differential equations
def euler_method(func, initial_conditions, beta, theta, g, y0, t_max, dt):
    t_values = np.arange(0, t_max + dt, dt)
    num_points = len(t_values)
    
    A_values = np.zeros(num_points)
    p_values = np.zeros(num_points)
    y_values = np.zeros(num_points)
    
    A_values[0], p_values[0], y_values[0] = initial_conditions
    
    for i in range(1, num_points):
        dA, dp, dy = func(t_values[i - 1], A_values[i - 1], p_values[i - 1], y_values[i - 1], beta, theta, g, y0)
        A_values[i] = A_values[i - 1] + dt * dA
        p_values[i] = p_values[i - 1] + dt * dp
        y_values[i] = y_values[i - 1] + dt * dy
    
    return t_values, A_values, p_values, y_values

# Set initial conditions
initial_conditions = [1, 0.5, 1.5]
beta = 0.5
theta = 3
g = 0.01
y0 = 1
t_max = 50
dt = 0.1

# Solve the system using Euler's method
t_values, A_values, p_values, y_values = euler_method(equations, initial_conditions, beta, theta, g, y0, t_max, dt)

# Print values at each time point
for i, t in enumerate(t_values):
    print(f"t={t:.2f}: A={A_values[i]:.4f}, p={p_values[i]:.4f}, y={y_values[i]:.4f}")

plt.figure(figsize=(10, 6))

# Plot the results with legend and specified labels
plt.plot(t_values, A_values, label='A(t)')
plt.plot(t_values, p_values, label='p(t)')
plt.plot(t_values, y_values, label='y(t)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.ylim(0, 10)
plt.legend()
plt.title('Industrial Revolution (Model 3)')
initial_conditions_text = f"Parameters:\nβ = {beta}\nθ = {theta}\ng = {g}\ny₀ = {y0}\np(0) = {initial_conditions[1]}\nA(0) = {initial_conditions[0]}\ny(0) = {initial_conditions[2]}"
plt.text(1.03, 0.5, initial_conditions_text, bbox=dict(facecolor='white', alpha=0.7), transform=plt.gca().transAxes)
plt.tight_layout()

plt.savefig('y_05.png', bbox_inches='tight')


plt.show()
