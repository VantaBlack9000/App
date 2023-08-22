#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import time
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from sko.GA import GA_TSP
from py2opt.routefinder import RouteFinder
import os

# Optimal Parameters found by grid search
node_sizes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
max_iter = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
pop_size = [50, 50, 50, 200, 50, 50, 200, 100, 250, 200, 250]
prob_mut = [0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.75, 0.5, 0.75, 0.25, 0.25]

def two_opt(cities_names, dist_mat):
    # Create a RouteFinder object
    route_finder = RouteFinder(dist_mat, cities_names, iterations = 1000)
    # Find the best route using 2-opt algorithm
    best_distance, best_route = route_finder.solve()

    # Append the first node to the end of the route to make it circular
    best_route.append(best_route[0])

    # Calculate the total distance of the circular tour
    total_distance = best_distance + dist_mat[best_route[-2]][best_route[-1]]

    return best_route, total_distance

# Function to calculate total distance of a tour
def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

# Genetic Algorithm with 2-opt improvement
def ga_with_2opt_improvement(num_points, max_iter, size_pop, prob_mut, dist_mat, cities_names):
    start_time = time.time()
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=size_pop, max_iter=max_iter, prob_mut=prob_mut)
    best_points, best_distance = ga_tsp.run()
    best_route, improved_distance = two_opt(cities_names, dist_mat)
    end_time = time.time()
    total_time = end_time - start_time
    return improved_distance, total_time


# Lists to store results for plotting
exact_lengths = []
ga_lengths = []
exact_times = []
ga_times = []

for i, num_points in enumerate(node_sizes):  # Use enumerate to get the index 'i'
    # Generate random points and calculate distance matrix
    points_coordinate = np.random.rand(num_points, 2)
    distance_matrix = np.linalg.norm(points_coordinate[:, np.newaxis] - points_coordinate, axis=2)
    cities_names = list(range(num_points))  # Names of cities for RouteFinder

    # Genetic Algorithm
    ga_solution_length, time_ga = ga_with_2opt_improvement(num_points, max_iter[i], pop_size[i], prob_mut[i], distance_matrix, cities_names)

    # Exact solution using python-tsp library
    start_time_exact = time.time()
    permutation, exact_solution_length = solve_tsp_dynamic_programming(distance_matrix)
    end_time_exact = time.time()
    time_exact = end_time_exact - start_time_exact

    # Store results
    exact_lengths.append(exact_solution_length)
    ga_lengths.append(ga_solution_length)
    exact_times.append(time_exact)
    ga_times.append(time_ga)

    # Calculate percentage difference
    percentage_difference = ((exact_solution_length - ga_solution_length) / exact_solution_length) * 100

    print(f"Number of Points: {num_points}")
    print(f"Exact Solution Length: {exact_solution_length:.2f}")
    print(f"GA Solution Length: {ga_solution_length:.2f}")
    print(f"Percentage Difference: {percentage_difference:.2f}%")
    print(f"Exact Solution Time: {time_exact:.4f} seconds")
    print(f"GA Solution Time: {time_ga:.4f} seconds\n")

# Create a plot to compare tour lengths
plt.figure(figsize=(10, 6))
plt.plot(node_sizes, exact_lengths, marker='o', label='Exact Solution')
plt.plot(node_sizes, ga_lengths, marker='o', label='GA Solution')
plt.xlabel('Number of Nodes')
plt.ylabel('Tour Length')
plt.title('Comparison of Exact Solution and GA Solution')
plt.legend(["Exact", "GA"])
plt.grid(True)
plt.savefig(r'C:\Users\Timmy Gerlach\Documents\Uni\Master\Masterarbeit\tour_lengths_comparison.png', dpi=300)
plt.show()

# Create a plot to compare running times
plt.figure(figsize=(10, 6))
plt.plot(node_sizes, exact_times, marker='o', label='Exact Solution')
plt.plot(node_sizes, ga_times, marker='o', label='GA Solution')
plt.xlabel('Number of Nodes')
plt.ylabel('Running Time (seconds)')
plt.title('Comparison of Running Times: Exact Solution vs. GA Solution')
plt.legend(["Exact", "GA"])
plt.grid(True)
plt.savefig(r'C:\Users\Timmy Gerlach\Documents\Uni\Master\Masterarbeit\running_times_comparison.png', dpi=300)
plt.show()


# In[17]:


import matplotlib.pyplot as plt

# Data (same as before)
node_sizes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
dp_lengths = [2.80, 3.57, 3.13, 3.25, 2.85, 3.43, 3.86, 3.65, 3.49, 3.76, 3.83]
ga_lengths = [3.13, 3.69, 3.13, 3.34, 3.01, 3.60, 3.96, 4.09, 4.04, 3.98, 4.35]
dp_times = [0.0111, 0.0291, 0.0673, 0.1784, 0.3973, 0.9783, 2.3835, 11.4872, 15.2746, 33.7304, 73.7538]
ga_times = [0.5150, 0.5400, 0.6235, 1.0419, 0.8013, 0.8840, 1.4424, 1.2856, 1.7491, 1.8273, 1.9884]

# Function to handle overlapping labels
def annotate_with_no_overlap(ax, x, y, labels, offset_points=(0, 5), condition=None):
    last_y = None
    for xi, yi, label in zip(x, y, labels):
        if (condition is None or condition(xi)) and (last_y is None or abs(yi - last_y) > 0.02):
            ax.annotate(f"{label:.2f}", (xi, yi), textcoords="offset points", xytext=offset_points, ha='center')
            last_y = yi

# First scatter plot: Tour Lengths Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(node_sizes, dp_lengths, label='Dynamic Programming', marker='o')
plt.scatter(node_sizes, ga_lengths, label='GA Solution', marker='x')
plt.xlabel('Node Sizes')
plt.ylabel('Tour Lengths')
plt.title('Comparison of Tour Lengths')
plt.grid(True)
plt.legend()

annotate_with_no_overlap(plt.gca(), node_sizes, dp_lengths, dp_lengths)
annotate_with_no_overlap(plt.gca(), node_sizes, ga_lengths, ga_lengths)

# Second scatter plot: Solution Times Comparison
plt.subplot(1, 2, 2)
plt.scatter(node_sizes, dp_times, label='Dynamic Programming', marker='o')
plt.scatter(node_sizes, ga_times, label='GA Solution', marker='x')
plt.xlabel('Node Sizes')
plt.ylabel('Solution Times (seconds)')
plt.title('Comparison of Solution Times')
plt.grid(True)
plt.legend()

annotate_with_no_overlap(plt.gca(), node_sizes, dp_times, dp_times, condition=lambda x: x > 16)
annotate_with_no_overlap(plt.gca(), node_sizes, ga_times, ga_times, condition=lambda x: x > 16)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(r'C:\Users\Timmy Gerlach\Documents\Uni\Master\Masterarbeit\running_times_comparison.png', dpi=300)
plt.show()


# In[ ]:




