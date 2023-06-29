from flask import Blueprint, render_template, render_template_string, request, make_response, session, current_app, flash, jsonify, redirect, url_for, send_file, Response
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sko.ACA import ACA_TSP
from sko.GA import GA_TSP
from sko.SA import SA_TSP
import io
import base64
import time
import folium
import os
from werkzeug.utils import secure_filename
import openrouteservice as ors
import requests
import geopy as gp
from geopy.geocoders import Nominatim
import geopandas as gpd
import json
import gpxpy
import gpxpy.gpx
import random
from py2opt.routefinder import RouteFinder

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
    start_time_sa = time.time()

    sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)

    best_points, best_distance = sa_tsp.run()
    print(best_points, best_distance, cal_total_distance(best_points))

    end_time_sa = time.time()
    total_time_sa = end_time_sa - start_time_sa

    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(sa_tsp.best_y_history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Distance")
    ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    plt.title("SA Output & Performance")
    plt.savefig("static/pictures/sa.png")
    best_distance_sa= best_distance
    best_points_sa = best_points

    #return frontend and variables
    return render_template("calculator.html", num_points=num_points, max_iter=max_iter, size_pop=size_pop, prob_mut=prob_mut, plot_url_aco="static/pictures/aco.png", plot_url_ga="static/pictures/ga.png", plot_url_sa="static/pictures/sa.png", total_time_aco=total_time_aco, total_time_ga=total_time_ga, total_time_sa=total_time_sa, best_distance_aco=best_distance_aco, best_distance_ga=best_distance_ga, best_distance_sa=best_distance_sa)

#initialize folium map object
m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
m.get_root().width = "800px"
m.get_root().height = "600px"


#View and method for uploading and diplaying the map
@views.route("/csv-calculator/", methods=["POST", "GET"])
def upload_csv():
    iframe = m.get_root()._repr_html_()
    if request.method == 'POST':
        uploaded_df = request.files['uploaded-file']
        data_filename = secure_filename(uploaded_df.filename)
        uploaded_df.save(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('csv_calc.html', iframe = iframe)
    
    else:
        return render_template("csv_calc.html", iframe=iframe)

#view and method for showing data table in new tab
@views.route("/csv-calculator-data/", methods=("POST","GET"))
def show_data():
    iframe = m.get_root()._repr_html_()
    data_file_path = session.get("uploaded_data_file_path", None)
    uploaded_df = pd.read_csv(data_file_path, delimiter=";")
    uploaded_df_html = uploaded_df.to_html()
    return render_template('csv_calc.html', data_var = uploaded_df_html, iframe=iframe)

#view and method for plotting the provided coords
@views.route("/plotted-data/", methods=["POST", "GET"])
def plot_csv():

    data_file_path = session.get("uploaded_data_file_path", None)
    coords = pd.read_csv(data_file_path, delimiter=";")

    if coords.empty:
        # Handle the case where there are no coordinates
        return render_template("csv_calc.html", iframe=None)

    #converting adress to geo coords
    if coords['lat'].isnull().any() and coords['long'].isnull().any():
        locator = Nominatim(user_agent="TSP_application_thesis")

        for index, row in coords.iterrows():
            address = f"{row['number']} ,{row['street']}, {row['city']}, {row['country']}"
            location = locator.geocode(address)
            coords.at[index, 'lat'] = location.latitude
            coords.at[index, 'long'] = location.longitude

        if coords["lat"].dtype == float and coords["long"].dtype == float:
            lat = coords["lat"].tolist()
            long = coords["long"].tolist()

        else: 
            lat = coords["lat"].astype(float).tolist()
            long = coords["long"].astype(float).tolist()

    #if coordinates were provided and are in float type
    elif coords["lat"].dtype == float and coords["long"].dtype == float:
        lat = coords["lat"].tolist()
        long = coords["long"].tolist()

    #if coordinates were provided but are not in float type 
    else:
        lat = coords["lat"].astype(float).tolist()
        long = coords["long"].astype(float).tolist()

    coords["lat"] = coords["lat"].round(6)
    coords["long"] = coords["long"].round(6)

    json_coords = coords.to_json()
    session["json_coords"] = json_coords

    points = []
    for i in range(len(lat)):
        points.append([lat[i], long[i]])

    #openrouteservice only takes in long, lat format
    points_ors = []
    for i in range(len(lat)):
        points_ors.append([long[i], lat[i]])

    #defining the top value of the table as starting point
    starting_point_coords = points[0]

    #plotting a default route
    response = client.directions(coordinates = points_ors, profile = "driving-car", format="geojson")
    route_coords = response["features"][0]["geometry"]["coordinates"]

    #convert back to folium lat, long format
    route_coords = [[coord[1], coord[0]] for coord in route_coords]

    marker_group = folium.FeatureGroup(name = "CSV Data")
    starting_point = folium.Marker(location = starting_point_coords, icon = folium.Icon(color="red"))
    for index, row in coords.iterrows():
        if index != 0: 
            lat = row["lat"]
            long = row["long"]
            marker = folium.Marker(location = [lat, long])
            marker_group.add_child(marker)

    #m.add_child(default_route)
    m.add_child(starting_point)
    m.add_child(marker_group)

    iframe = m.get_root()._repr_html_()
    return render_template("csv_calc.html", iframe=iframe)

@views.route("/add-customers/", methods=["POST", "GET"])
def manually_add_customers():
    iframe = m.get_root()._repr_html_()
    if request.method == "POST":
        customer_name = request.form.get("customer_name")
        house_number = request.form.get("house_number")
        street_name = request.form.get("street_name")
        city = request.form.get("city")
        country = request.form.get("country")
        initial_lat = None
        initial_long = None

        # Add the new lead to the 'coords' DataFrame
        #new_lead= None
        new_lead = pd.DataFrame(
            [[customer_name, house_number, street_name, city, country, initial_lat, initial_long]],
            columns=["name", "number", "street", "city", "country", "lat", "long"]
        )
        coords = pd.read_json(session.get("json_coords"))
        coords = pd.concat([coords, new_lead], ignore_index=True)

        # Reset the index of 'coords' DataFrame
        coords = coords.reset_index(drop=True)

        # Update the 'json_coords' in the session
        session["json_coords"] = coords.to_json()

        # Save the updated DataFrame as a CSV file, overwriting the existing file

        data_file_path = session.get("uploaded_data_file_path")
        if data_file_path:
            coords.to_csv(data_file_path, sep=";", index=False)

        # Redirect to the same page to prevent form resubmission
        return redirect(url_for("views.manually_add_customers"))

    return render_template("csv_calc.html", iframe=iframe)

@views.route("/distances-csv/", methods=["POST", "GET"])
def calculate_csv_distance():
    
    max_iter = 200
    size_pop = 50
    prob_mut = 1

    data_file_path = session.get("uploaded_data_file_path", None)
    #coords = pd.read_csv(data_file_path, delimiter=";")
    json_coords = session.get("json_coords")
    coords = pd.read_json(json_coords)

    best_params_ga = pd.read_csv(r"C:\Users\Timmy Gerlach\Documents\Uni\Master\Masterarbeit\App\static\files\best_params_grid_search.csv", sep = ";")

    if len(coords) in best_params_ga["problem_size"].values:
        row_index = best_params_ga.loc[best_params_ga["problem_size"] == len(coords)].index[0]
        max_iter = best_params_ga.loc[row_index, "max_iter"]
        size_pop = best_params_ga.loc[row_index, "pop_size"]
        prob_mut = best_params_ga.loc[row_index, "prob_mut"]

    #the number of points is given by the lengths of the coords data frame
    num_points = len(coords.index)

    #coordinates in lat, long format for folium
    points_coordinate = np.array(coords[["lat", "long"]])

    #coordinates in long, lat format for openrouteservice
    points_coordinate_ors = np.array(coords[["long","lat"]])

    # Get the selected vehicle from the form
    selected_vehicle = request.form.get("Type of Locomotion")

    # Set the profile based on the selected vehicle
    if selected_vehicle == "car":
        profile = "driving-car"
    elif selected_vehicle == "walking":
        profile = "foot-walking"
    elif selected_vehicle == "bike":
        profile = "cycling-regular"
    else:
        # Default to "driving-car" if no vehicle is selected
        profile = "driving-car"

    #calculation of distance matrix
    response = client.distance_matrix(locations=points_coordinate_ors.tolist(), metrics=["distance"], profile=profile)
    distance_matrix = np.array(response["distances"])
    distance_matrix_km = distance_matrix / 1000

    starting_point = points_coordinate[0]

    #def cal_total_distance(routine):
    #    num_points, = routine.shape
    #    return sum([distance_matrix_km[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    def cal_total_distance(routine):
        num_points, = routine.shape
        total_distance = 0

        for i in range(num_points):
            total_distance += distance_matrix_km[
                routine[i % num_points], routine[(i + 1) % num_points]
            ]

        return total_distance
    
    
    def two_opt(route):
        # Create a RouteFinder object
        route_finder = RouteFinder(route)

        # Find the best route using 2-opt algorithm
        best_route, best_distance = route_finder.solve()

        return best_route, best_distance
    
    #genetic algorithm
    start_time_ga =time.time()
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
    best_points, best_distance = ga_tsp.run()
    end_time_ga= time.time()
    total_time_ga= end_time_ga - start_time_ga
    best_distance_ga = best_distance[0]
    
    #rearrangement of the points_coordinate variables to make them store the best tours coordinates
    best_tour_ga_ors = points_coordinate_ors[np.argsort(best_points)]

    #conveting to lists
    list_ga_ors = best_tour_ga_ors.tolist()

    #Applying 2-opt optimization
    optimized_route = two_opt(list_ga_ors)

    #plotting the best route calculated by the ga
    response_ga = client.directions(coordinates = optimized_route, profile = profile, format="geojson")
    route_coords_ga = response_ga["features"][0]["geometry"]["coordinates"]

    route_coords_ga = [[coord[1], coord[0]] for coord in route_coords_ga]

    best_ga_route = folium.PolyLine(locations=route_coords_ga, color="red")

    m.add_child(best_ga_route)

    iframe = m.get_root()._repr_html_()

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Add points to the track segment
    for coord in best_tour_ga_ors:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=coord[1], longitude=coord[0]))

 # Store GPX data as a session variable
    session["gpx_data"] = gpx.to_xml()

    return render_template("csv_calc.html", iframe=iframe, max_iter=max_iter, num_points=num_points, size_pop=size_pop, prob_mut=prob_mut, total_time_ga=total_time_ga, best_distance_ga=best_distance_ga )

@views.route("/download-gpx/", methods=["GET"])
def download_gpx():
    #iframe = m.get_root()._repr_html_()
    gpx_data = session.get("gpx_data")
    if gpx_data:
        # Set the appropriate headers for file download
        headers = {
            "Content-Disposition": "attachment; filename=route.gpx",
            "Content-Type": "application/gpx+xml",
        }
        return Response(gpx_data, headers=headers)

@views.route("/contact/")
def contact():
    return render_template("contact.html")

@views.route("/sources/")
def sources():
    return render_template("sources.html")



