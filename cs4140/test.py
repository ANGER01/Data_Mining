import matplotlib.pyplot as plt

# Shared x-coordinates
x = [1, 2, 3, 4, 5]

# Different y-coordinates for each line
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]
y3 = [5, 10, 15, 20, 25]

# Plot each line
plt.plot(x, y1, label='Line 1', marker='o', linestyle='-', color='b')
plt.plot(x, y2, label='Line 2', marker='s', linestyle='--', color='g')
plt.plot(x, y3, label='Line 3', marker='^', linestyle='-.', color='r')

# Add labels, title, and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of 3 Lines with Same X')
plt.legend()

# Optional: Add a grid
plt.grid()

# Show the plot
plt.show()
