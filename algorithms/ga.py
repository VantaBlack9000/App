import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from general import num_points, points_coordinate, distance_matrix, cal_total_distance, size_pop, max_iter, prob_mut


from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
best_points, best_distance = ga_tsp.run()

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()