import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)  # Setting a seed for reproducibility
x_coordinates = np.random.rand(12) * 10  # Random x coordinates between 0 and 10
y_coordinates = np.random.rand(12) * 10  # Random y coordinates between 0 and 10

np.random.seed(3)
all_indices = list(range(12))
np.random.shuffle(all_indices)

x_indices = all_indices[:5]
o_indices = all_indices[4:9]


# Mark 4 points as 'x'
# x_indices = [0, 3, 6, 9]
x_x = [x_coordinates[i] for i in x_indices]
x_y = [y_coordinates[i] for i in x_indices]

# Mark 4 points as 'o'
# o_indices = [1, 4, 7, 10]
o_x = [x_coordinates[i] for i in o_indices]
o_y = [y_coordinates[i] for i in o_indices]


fig = plt.figure()
plt.rcParams['figure.figsize'] = [6, 4]

# Plot the points
plt.scatter(x_coordinates, y_coordinates, marker='.', label='Unpicked', color="black", s=100)
plt.scatter(x_x, x_y, marker='v', label='Run 1', color="orange", s=150)
plt.scatter(o_x, o_y, marker='^', label='Run 2', color="blue", s=150)

# Add labels and legend
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])

plt.legend()

# Show the plot
plt.grid()
plt.savefig("Fig/NDSource/Restart.pdf", bbox_inches='tight')

plt.show()



