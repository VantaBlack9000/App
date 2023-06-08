from flask import Blueprint, render_template, render_template_string, request, make_response, session, current_app, flash, jsonify
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
from sko.GA import GA_TSP
import io
import base64
import time
import folium
import os
from werkzeug.utils import secure_filename
import openrouteservice as ors
import requests

API_KEY = "5b3ce3597851110001cf6248c09bb9f319ff486dbaae400d6f00a30d"
client = ors.Client(key=API_KEY)

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
    start_time_aco = time.time()

    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
               size_pop = size_pop, max_iter=max_iter,
              distance_matrix=distance_matrix)

    best_x, best_y = aca.run()

    end_time_aco = time.time()
    total_time_aco = end_time_aco - start_time_aco

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.title("ACO Output & Performance")
    plt.savefig("static/pictures/aco.png")
    best_distance_aco = best_y
    
    #genetic algorithm
    start_time_ga =time.time()

    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
    best_points, best_distance = ga_tsp.run()

    end_time_ga= time.time()
    total_time_ga= end_time_ga - start_time_ga

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    ax[1].plot(ga_tsp.generation_best_Y)
    plt.title("GA Output & Performance")
    plt.savefig("static/pictures/ga.png")
    best_distance_ga = best_distance[0]

    #simulated annealing algorithm

    #return frontend and variables
    return render_template("calculator.html", num_points=num_points, max_iter=max_iter, size_pop=size_pop, prob_mut=prob_mut, plot_url_aco="static/pictures/aco.png", plot_url_ga="static/pictures/ga.png", total_time_aco=total_time_aco, total_time_ga=total_time_ga, best_distance_aco=best_distance_aco, best_distance_ga=best_distance_ga)

#initialize folium map object
m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
m.get_root().width = "800px"
m.get_root().height = "600px"


#View and method for uploading and diplaying the map
@views.route("/csv-calculator/", methods=["POST", "GET"])
def upload_csv():

    if request.method == 'POST':
        uploaded_df = request.files['uploaded-file']
        data_filename = secure_filename(uploaded_df.filename)
        uploaded_df.save(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename)
        return render_template('csv_calc.html')
    
    else:
        return render_template("csv_calc.html")

#view and method for showing data table in new tab
@views.route("/csv-calculator-data/", methods=("POST","GET"))
def show_data():
    data_file_path = session.get("uploaded_data_file_path", None)
    uploaded_df = pd.read_csv(data_file_path, delimiter=";")
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var = uploaded_df_html)

#view and method for plotting the provided coords
@views.route("/plotted-data/", methods=["POST", "GET"])
def plot_csv():

    data_file_path = session.get("uploaded_data_file_path", None)
    coords = pd.read_csv(data_file_path, delimiter=";")
    
    if coords["lat"].dtype == float and coords["long"].dtype == float:
        lat = coords["lat"].tolist()
        long = coords["long"].tolist()
    else:
        lat = coords["lat"].astype(float).tolist()
        long = coords["long"].astype(float).tolist()

    points = []
    for i in range(len(lat)):
        points.append([lat[i], long[i]])

    #openrouteservice only takes in long, lat format
    points_ors = []
    for i in range(len(lat)):
        points_ors.append([long[i], lat[i]])

    starting_point_coords = points[0]

    #plotting a default route
    response = client.directions(coordinates = points_ors, profile = "driving-car", format="geojson")
    route_coords = response["features"][0]["geometry"]["coordinates"]

    #convert back to folium lat, long format
    route_coords = [[coord[1], coord[0]] for coord in route_coords]

    default_route = folium.PolyLine(locations=route_coords, color="blue")
    marker_group = folium.FeatureGroup(name = "CSV Data")
    starting_point = folium.Marker(location = starting_point_coords, icon = folium.Icon(color="red"))
    for index, row in coords.iterrows():
        if index != 0: 
            lat = row["lat"]
            long = row["long"]
            marker = folium.Marker(location = [lat, long])
            marker_group.add_child(marker)

    m.add_child(default_route)
    m.add_child(starting_point)
    m.add_child(marker_group)

    iframe = m.get_root()._repr_html_()
    return render_template("csv_calc.html", iframe=iframe)

@views.route("/distances-csv/", methods=["POST", "GET"])
def calculate_csv_distance():

    if request.method == "POST":
        max_iter = int(request.form["max_iter"])
        size_pop = int(request.form["size_pop"])
        prob_mut = int(request.form["prob_mut"])
    else:
        max_iter = 200
        size_pop = 50
        prob_mut = 1

    data_file_path = session.get("uploaded_data_file_path", None)
    coords = pd.read_csv(data_file_path, delimiter=";")

    #the number of points is given by the lengths of the coords data frame
    num_points = len(coords.index)

    #coordinates in lat, long format for folium
    points_coordinate = np.array(coords[["lat", "long"]])

    #coordinates in long, lat format for openrouteservice
    points_coordinate_ors = np.array(coords[["long","lat"]])

    #calculation of distance matrix
    response = client.distance_matrix(locations=points_coordinate_ors.tolist(), metrics=["distance"], profile="driving-car")
    distance_matrix = np.array(response["distances"])
    distance_matrix_km = distance_matrix / 1000
    #distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    starting_point = points_coordinate[0]

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix_km[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    
    #ant colony optimization
    start_time_aco = time.time()
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
               size_pop = size_pop, max_iter=max_iter,
              distance_matrix=distance_matrix_km)
    best_x, best_y = aca.run()
    end_time_aco = time.time()
    total_time_aco = end_time_aco - start_time_aco
    best_distance_aco = best_y

    #genetic algorithm
    start_time_ga =time.time()
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
    best_points, best_distance = ga_tsp.run()
    end_time_ga= time.time()
    total_time_ga= end_time_ga - start_time_ga
    best_distance_ga = best_distance[0]

    #rearrangement of the points_coordinate variables to make them store the best tours coordinates
    best_tour_aco_ors = points_coordinate_ors[np.argsort(best_x)]
    best_tour_ga_ors = points_coordinate_ors[np.argsort(best_points)]

    #conveting to lists
    list_aco_ors = best_tour_aco_ors.tolist()
    list_ga_ors = best_tour_ga_ors.tolist()

    #plotting the best routes
    response_aco = client.directions(coordinates = list_aco_ors, profile = "driving-car", format="geojson")
    response_ga = client.directions(coordinates = list_ga_ors, profile = "driving-car", format="geojson")
    route_coords_aco = response_aco["features"][0]["geometry"]["coordinates"]
    route_coords_ga = response_ga["features"][0]["geometry"]["coordinates"]

    route_coords_aco = [[coord[1], coord[0]] for coord in route_coords_aco]
    route_coords_ga = [[coord[1], coord[0]] for coord in route_coords_ga]

    best_aco_route = folium.PolyLine(locations=route_coords_aco, color="blue")
    best_ga_route = folium.PolyLine(locations=route_coords_ga, color="red")

    m.add_child(best_aco_route)
    m.add_child(best_ga_route)

    iframe = m.get_root()._repr_html_()

    return render_template("csv_calc.html", iframe=iframe, max_iter=max_iter, num_points=num_points, size_pop=size_pop, prob_mut=prob_mut, total_time_aco=total_time_aco, total_time_ga=total_time_ga, best_distance_aco=best_distance_aco, best_distance_ga=best_distance_ga )

@views.route("/click-calculator")
def click_calculator():
    

    iframe = m.get_root()._repr_html_()
    return render_template("click_calculator.html", iframe = iframe)



@views.route("/contact/")
def contact():
    return render_template("contact.html")

@views.route("/sources/")
def sources():
    return render_template("sources.html")