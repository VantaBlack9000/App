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
from flask_caching import Cache
import random
import operator
from functools import reduce

API_KEY = "5b3ce3597851110001cf6248c09bb9f319ff486dbaae400d6f00a30d"
client = ors.Client(key=API_KEY)

#MODULES START HERE########################################################################################################################
def remove_files_in_folder():
    folder_path = "C:\\Users\\Timmy Gerlach\\Documents\\Uni\\Master\\Masterarbeit\\App\\static\\map_csv_files"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")



#ROUTES START HERE#########################################################################################################################
views = Blueprint(__name__, "views")

@views.route("/")
def home():
    session.clear()
    remove_files_in_folder()
    return render_template("home.html")

@views.route("/about/")
def about():
    session.clear()
    return render_template("about.html")

#initialize folium map object
m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
m.get_root().width = "100%"
m.get_root().height = "600px"


#View and method for uploading and diplaying the map
@views.route("/csv-calculator/", methods=["POST", "GET"])
def upload_csv():

    #clear the session variables
    session.clear()

    #clear the upload folder form old data
    remove_files_in_folder()

    #initialize map element
    iframe = m.get_root()._repr_html_()
    
    if request.method == 'POST':
        uploaded_df = request.files['uploaded-file']
        data_filename = secure_filename(uploaded_df.filename)
        uploaded_df.save(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename)

        try:
            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        except pd.errors.EmptyDataError:
            # Handle the EmptyDataError and display a warning message
            warning_message = "Warning: The CSV file is empty."
            return render_template('csv_calc.html', iframe=iframe, warning_message=warning_message)

        # Check if the DataFrame is empty or has more than 50 rows (excluding column names)
        if df.empty or (len(df.index) > 50):
            warning_message = "Warning: The CSV file is either empty or has more than 50 rows (excluding column names)."
        else:
            warning_message = None
            session["file_uploaded"] = True

        return render_template('csv_calc.html', iframe=iframe, warning_message=warning_message)
    
    else:
        return render_template("csv_calc.html", iframe=iframe)

#view and method for showing data table in new tab
@views.route("/csv-calculator-data/", methods=("POST","GET"))
def show_data():
    iframe = m.get_root()._repr_html_()
    data_file_path = session.get("uploaded_data_file_path", None)
    try:
        uploaded_df = pd.read_csv(data_file_path, delimiter=";")
    except pd.errors.EmptyDataError:
            # Handle the EmptyDataError and display a warning message
            warning_message_show = "Warning: No data uploaded yet."
            return render_template('csv_calc.html', iframe=iframe, warning_message_show=warning_message_show)

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

    global marker_group
    marker_group = folium.FeatureGroup(name = "CSV Data")
    for index, row in coords.iterrows():
        lat = row["lat"]
        long = row["long"]
        marker = folium.Marker(location = [lat, long])
        marker_group.add_child(marker)

    m.get_root().height = "600px"
    m.add_child(marker_group)
    iframe = m.get_root()._repr_html_()
    return render_template("csv_calc.html", iframe=iframe, m=m)

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

        if session.get("json_coords"):
            coords = pd.read_json("json_coords")
        else:
            coords = pd.DataFrame(columns=["name", "number", "street", "city", "country", "lat", "long"])

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
    if session.get("file_uploaded") == True:
        max_iter = 200
        size_pop = 50
        prob_mut = 1

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

        #coordinates in long, lat format for openrouteservice
        points_coordinate_ors = np.array(coords[["long","lat"]])
        points_coordinate_ors = np.concatenate([points_coordinate_ors, [points_coordinate_ors[0]]])

        # Get the selected vehicle from the form
        selected_vehicle = request.form.get("Type of Locomotion")

        # Set the profile based on the selected vehicle
        if selected_vehicle == "car":
            profile = "driving-car"
            session["profile"] = profile
        elif selected_vehicle == "walking":
            profile = "foot-walking"
            session["profile"] = profile
        elif selected_vehicle == "bike":
            profile = "cycling-regular"
            session["profile"] = profile
        else:
            # Default to "driving-car" if no vehicle is selected
            profile = "driving-car"
            session["profile"] = profile

        #calculation of distance matrix
        response = client.distance_matrix(locations=points_coordinate_ors.tolist(), metrics=["distance"], profile=profile)
        distance_matrix = np.array(response["distances"])
        distance_matrix_km = distance_matrix / 1000

        def cal_total_distance(routine):
            num_points, = routine.shape
            total_distance = 0

            for i in range(num_points):
                total_distance += distance_matrix_km[
                    routine[i % num_points], routine[(i + 1) % num_points]
                ]

            return total_distance

        #genetic algorithm
        ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
        best_points, best_distance = ga_tsp.run()
        best_distance_ga = best_distance[0]
        best_points = np.concatenate([best_points, [best_points[0]]])

        #rearrangement of the points_coordinate variables to make them store the best tours coordinates
        best_tour_ga_ors = points_coordinate_ors[np.argsort(best_points)]

        #conveting to lists
        list_ga_ors = best_tour_ga_ors.tolist()

        updated_list_ga_ors = list_ga_ors 
        session["final_coordinates"] = updated_list_ga_ors

        #plotting the best route calculated by the ga
        response_ga = client.directions(coordinates = updated_list_ga_ors, profile = profile, format="geojson")
        
        route_coords_ga = response_ga["features"][0]["geometry"]["coordinates"]
        route_coords_ga = [[coord[1], coord[0]] for coord in route_coords_ga]

        waypoints = list(dict.fromkeys(reduce(operator.concat, list(map(lambda step: step["way_points"], response_ga["features"][0]["properties"]["segments"][0]["steps"] )))))
        directions_ga = folium.PolyLine(locations=[list(reversed(response_ga["features"][0]["geometry"]["coordinates"][index])) for index in waypoints], color = "green")
        best_ga_route = folium.PolyLine(locations=route_coords_ga, color="red")

        # Extract waypoints from the response_ga object
        waypoints = []
        for step in response_ga["features"][0]["properties"]["segments"][0]["steps"]:
            waypoints.extend(step["way_points"])

        instructions = []
        for step in response_ga["features"][0]["properties"]["segments"][0]["steps"]:
            instructions.extend(step["instruction"])
            
        # Add markers to the map for each waypoint
        for i, step in enumerate(response_ga["features"][0]["properties"]["segments"][0]["steps"]):
            lat, lon = reversed(response_ga["features"][0]["geometry"]["coordinates"][waypoints[i]])
            instruction = step["instruction"]
            marker_coords = [lat, lon]
            popup_text = f"Coordinates: {lat}, {lon}<br>Instruction: {instruction}"
            folium.Marker(location=marker_coords, icon=folium.Icon(color='green'), popup=popup_text).add_to(m)

        #m.add_child(directions_ga)
        m.add_child(best_ga_route)
        m.add_child(marker_group)
        iframe = m.get_root()._repr_html_()

        return render_template("csv_calc.html", iframe=iframe, max_iter=max_iter, num_points=num_points, size_pop=size_pop, prob_mut=prob_mut, best_distance_ga=best_distance_ga )

    else:
        iframe = m.get_root()._repr_html_()
        warning_message_calculator = "Please upload data"
        return render_template("csv_calc.html", iframe=iframe, warning_message_calculator=warning_message_calculator)

#View for retreiving the corresponding data as a gpx file for further implementation in tools and devices
@views.route('/download-gpx/', methods=['GET'])
def download_gpx():
    # Extract coordinates and other parameters from the request
    final_coordinates = session.get('final_coordinates')
    profile = session.get("profile")
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': API_KEY,  # Replace with your actual API key
        'Content-Type': 'application/json; charset=utf-8'
    }

    # Create the request body
    body = {"coordinates": final_coordinates}

    # Make the POST request to OpenRouteService API
    call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car/gpx', json=body, headers=headers)

    if call.status_code == 200:
        gpx_data = call.text
        headers = {
            'Content-Disposition': 'attachment; filename="route.gpx"',
            'Content-Type': 'application/gpx+xml'
        }
        return Response(gpx_data, headers=headers)
    else:
        return "Error: Unable to retrieve GPX data"


#section for the comparison of the algorithms. Use the route /calculator in the app for accessing it.
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
