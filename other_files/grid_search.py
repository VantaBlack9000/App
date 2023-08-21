# Import necessary libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.GA import GA_TSP
from scipy import spatial
from scipy.spatial import distance
from tqdm import tqdm
import time

# A function for generating random points, that takes in the problem size and return a Pandas Dataframe with columns lat and long
def generate_random_coordinates(num_points):
    points_coordinate = np.random.rand(num_points, 2)
    return pd.DataFrame(points_coordinate, columns=["lat", "lon"])

# The objective function taken from the sko Documentation example
def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

# The sample of the different parameters the grid serach should iterate through
problem_sizes = range(3, 51)
max_iters = [100, 200, 300]
size_pops = [50, 100, 150]
prob_muts = [0.1, 0.5, 1.0]

# Initiaizing a list for the results
results = []

# Initialize a total iterations variable fo ra tqdm bar to measure the progress of the operation
total_iterations = len(problem_sizes) * len(max_iters) * len(size_pops) * len(prob_muts)
pbar = tqdm(total=total_iterations)  # Initialize progress bar

# Begin the Grid Search
# Iterate through all problem sizes 
for problem_size in problem_sizes:

    # Generate a random problem for each problem size
    coords = generate_random_coordinates(problem_size)

    # Convert the DataFrame to a numpy array
    points_coordinate = coords[["lat", "lon"]].to_numpy()

    # Calculate the distance matrix. Taken from the SKO documentation
    distance_matrix = distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    best_result = None
    best_distance = float('inf')

    # Loop through the sample of given for the max iteration parameter
    for max_iter in max_iters:

        #Loop through the sample given for the population size parameter
        for size_pop in size_pops:

            # Loop through the sample given for the probability of mutation parameter
            for prob_mut in prob_muts:

                # Initialize starting time
                start_time_ga = time.time()

                # Initialize GA. Taken from SKO Documenattion and modified. 
                ga_tsp = GA_TSP(
                    func=cal_total_distance,
                    n_dim=problem_size,
                    size_pop=size_pop,
                    max_iter=max_iter,
                    prob_mut=prob_mut
                )

                # Run the GA
                best_points, best_distance_ga = ga_tsp.run()

                # Stop the timing and calculate the total time taken
                end_time_ga = time.time()
                total_time_ga = end_time_ga - start_time_ga

                # Check if the selected parameter enhanced the result
                if best_distance_ga < best_distance:
                    # Store the parameters for the best result
                    best_result = {
                        "problem_size": problem_size,
                        "max_iter": max_iter,
                        "size_pop": size_pop,
                        "prob_mut": prob_mut,
                        "best_distance_ga": best_distance_ga,
                        "calculation_time": total_time_ga
                    }

                    #Update the best distance variable
                    best_distance = best_distance_ga

                # Update the tqdm bar to visualize the progress
                pbar.update(1)  # Update progress bar

    results.append(best_result)

pbar.close()  # Close the progress bar

# Generate performance plots
x_ticks = [result["problem_size"] for result in results]
y_time = [result["total_time_ga"] for result in results]
y_distance = [result["best_distance_ga"] for result in results]

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(x_ticks, y_time, marker='o')
ax[0].set_xlabel("Problem Size")
ax[0].set_ylabel("Execution Time (s)")
ax[0].set_title("Execution Time vs. Problem Size")

ax[1].plot(x_ticks, y_distance, marker='o')
ax[1].set_xlabel("Problem Size")
ax[1].set_ylabel("Best Distance")
ax[1].set_title("Best Distance vs. Problem Size")

plt.tight_layout()
plt.show()

# Print out the resulting parameters
for result in results:
    print(f"Problem Size: {result['problem_size']}")
    print(f"Best Parameters: Max Iter: {result['max_iter']}, Size Pop: {result['size_pop']}, Prob Mut: {result['prob_mut']}")
    print(f"Best Distance: {result['best_distance_ga']}")
    print()