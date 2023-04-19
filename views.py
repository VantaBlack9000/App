from flask import Blueprint, render_template, request
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
from sko.GA import GA_TSP

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("home.html")

@views.route("/about/")
def about():
    return render_template("about.html")

@views.route("/algorithms/")
def algorithms():
    return render_template("algorithms.html")

@views.route("/calculator/", methods=["GET", "POST"])
def calculate_aco():

    if request.method == "POST":
        num_points = int(request.form["num_points"])
        max_iter = int(request.form["max_iter"])
        size_pop = int(request.form["size_pop"])
        prob_mut = int(request.form["prob_mut"])
    else:
        num_points = 10
        max_iter = 200
        size_pop = 50
        prob_mut = 1

    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    
    #ant colony optimization
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
               size_pop = size_pop, max_iter=max_iter,
              distance_matrix=distance_matrix)

    best_x, best_y = aca.run()

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.title("ACO Output & Performance")
    plt.savefig("C:/Users/Timmy Gerlach/Documents/Uni/Master/Masterarbeit/App/static/pictures/aco.png")
    
    #genetic algorithm
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
    best_points, best_distance = ga_tsp.run()

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    ax[1].plot(ga_tsp.generation_best_Y)
    plt.title("GA Output & Performance")
    plt.savefig("C:/Users/Timmy Gerlach/Documents/Uni/Master/Masterarbeit/App/static/pictures/ga.png")

    #return frontend and variables
    return render_template("calculator.html", num_points=num_points, max_iter=max_iter, size_pop=size_pop, prob_mut=prob_mut, plot_url="C:/Users/Timmy Gerlach/Documents/Uni/Master/Masterarbeit/App/static/pictures/aco.png")

@views.route("/contact/")
def contact():
    return render_template("contact.html")

@views.route("/sources/")
def sources():
    return render_template("sources.html")