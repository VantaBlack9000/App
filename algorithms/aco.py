import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from general import num_points, points_coordinate, distance_matrix, cal_total_distance, size_pop, max_iter

from sko.ACA import ACA_TSP

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              pop_size = size_pop, iterations=max_iter,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()