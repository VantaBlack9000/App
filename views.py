# Import necessary libraries and packages
from flask import Blueprint, render_template, request, session, current_app, redirect, url_for, Response
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from sko.ACA import ACA_TSP
from sko.GA import GA_TSP
from sko.SA import SA_TSP
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
from py2opt.routefinder import RouteFinder
from flask_caching import Cache
import random
import operator
from functools import reduce
from scipy.spatial import distance
from geopy.distance import geodesic

#IMPORTANT INITIAL VARIABLES###############################################################################################################
# The API key for the open route service API
API_KEY = "5b3ce3597851110001cf6248c09bb9f319ff486dbaae400d6f00a30d"

# Initialize the API
client = ors.Client(key=API_KEY)

#initialize folium map object
m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
m.get_root().width = "100%"
m.get_root().height = "600px"

# Define the views for the import in app.py. This has been done to achive a clear overview and seperation of app configuration and functionalities
views = Blueprint(__name__, "views")

#MODULES START HERE########################################################################################################################

# A function that removes all files in the upload folder so no data of other users is accessable
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

# View for displaying the home-page
@views.route("/")
def home():
    # Clear the User session to ensure a clean start of the application
    session.clear()

    # Clear the file in which uploaded files were stored priorily. The function is defined in the modules section
    remove_files_in_folder()

    # Return the template for the Home-page of the application
    return render_template("home.html")

# View for displaying the about-page
@views.route("/about/")
def about():

    # Clear the session of every user, every time he visits a new page
    session.clear()
    
    # Return the template for the About-page of the application
    return render_template("about.html")

# View and method for uploading and diplaying the map
@views.route("/csv-calculator/", methods=["POST", "GET"])
def upload_csv():

    # Clear the session variables
    session.clear()

    # Clear the upload folder form old data
    remove_files_in_folder()

    # Initialize a boolean that later on signilizes, if data was uploaded
    session["file_uploaded"] = False

    #initialize map element
    iframe = m.get_root()._repr_html_()
    
    # Check if the data is submitted 
    if request.method == 'POST':
        # Get the upload file from the POST request
        uploaded_df = request.files['uploaded-file']
        data_filename = secure_filename(uploaded_df.filename)

        # Save the uploaded file to the upload folder
        uploaded_df.save(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename)

        try:
            # Try to read the uploaded csv into a Pandas DataFrame
            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], data_filename))
        except pd.errors.EmptyDataError:
            # Handle the EmptyDataError and display a warning message (If the user uploaded an empty file)
            warning_message = "Warning: The CSV file is empty."
            return render_template('csv_calc.html', iframe=iframe, warning_message=warning_message)

        # Check if the DataFrame is empty or has more than 50 rows (excluding column names)
        if df.empty or (len(df.index) > 50):
            warning_message = "Warning: The CSV file is either empty or has more than 50 rows (excluding column names)."
        else:
            warning_message = None
            session["file_uploaded"] = True

        # Return the corresponding template and warning messages (if existent) 
        return render_template('csv_calc.html', iframe=iframe, warning_message=warning_message)
    
    else:
        # If no data is uploaded, this just return the initial template
        return render_template("csv_calc.html", iframe=iframe)

# View and method for showing data table in new tab
@views.route("/csv-calculator-data/", methods=("POST","GET"))
def show_data():

    # Store the map in an iframe to render it 
    iframe = m.get_root()._repr_html_()

    # Use the boolean created in in the upload_csv function to check if data has been uploaded
    if session.get("file_uploaded") == False:
        # If no data has been uploaded, a warning message is generated, passed to the HTML template and displayed to the user
        warning_message_show = "Warning: No data uploaded yet."
        return render_template('csv_calc.html', iframe=iframe, warning_message_show=warning_message_show)
    else:
        # If a file has been uploaded, the file path is retrieved from the session
        data_file_path = session.get("uploaded_data_file_path", None)
        try:
            # Attempting to read the data into a Pandas DataFrame
            uploaded_df = pd.read_csv(data_file_path, delimiter=";")
        except pd.errors.EmptyDataError:
                 # If the data uploaded is still empty due to an unecpected error earlier, a warning message is displayed
                warning_message_show = "Warning: No data uploaded yet."
                return render_template('csv_calc.html', iframe=iframe, warning_message_show=warning_message_show)

        # Convert the DataFrame to an HTMl representation, so it can be rendered and displayed on the page
        uploaded_df_html = uploaded_df.to_html()

        # Return teh corresponding template, together with the map and pass the HTML representation, to make use of it in the HTML files
        return render_template('csv_calc.html', data_var = uploaded_df_html, iframe=iframe)

# View for adding customers manually by typing in adresses 
@views.route("/add-customers/", methods=["POST", "GET"])
def manually_add_customers():
    # Render the map object again for display
    iframe = m.get_root()._repr_html_()

    # Check, if the requested method is a POST method (when submission button is clicked)
    if request.method == "POST":
        # Retrieve the data from the form, if the user tries to type in adresses manually
        customer_name = request.form.get("customer_name")
        house_number = request.form.get("house_number")
        street_name = request.form.get("street_name")
        city = request.form.get("city")
        country = request.form.get("country")
        initial_lat = None
        initial_long = None

        # If the user types in the data manually, a new Pandas DataFrame is initializes with the corresponding data inside
        new_lead = pd.DataFrame(
            [[customer_name, house_number, street_name, city, country, initial_lat, initial_long]],
            columns=["name", "number", "street", "city", "country", "lat", "long"]
        )

        # Retrieve or create a Pandas DataFrame for the coordinates in the session, if the user tries to add adresses to the ones he already uploaded
        if session.get("json_coords"):
            coords = pd.read_json(session.get("json_coords"))
        else:
            coords = pd.DataFrame(columns=["name", "number", "street", "city", "country", "lat", "long"])

        # Merge the new lead to the DataFrame with the already exiting coordinates
        coords = pd.concat([coords, new_lead], ignore_index=True)

        # Reset the index of 'coords' DataFrame
        coords = coords.reset_index(drop=True)

        # Update the 'json_coords' in the session
        session["json_coords"] = coords.to_json()

        # Save the updated DataFrame as a CSV file, overwriting the existing file
        session['uploaded_data_file_path'] = 'UPLOAD_FOLDER'
        data_file_path = session.get("uploaded_data_file_path", None)
        if data_file_path:
            coords.to_csv(data_file_path, sep=";", index=False)
        
        # Update the file_uploaded status to True, so it can later be used as a signal
        session["file_uploaded"] = True

        # Redirect to the same page to prevent form resubmission
        return redirect(url_for("views.manually_add_customers"))

    # Render and return the corresponding template as well as the map iframe
    return render_template("csv_calc.html", iframe=iframe)

# View and method for plotting the provided coords as markers on the map
@views.route("/plotted-data/", methods=["POST", "GET"])
def plot_csv():

    # Use the priorly created boolean to check if data has been uploaded
    if session.get("file_uploaded") == True:
        # If data has been uploaded, the file path is taken from the session
        data_file_path = session.get("uploaded_data_file_path", None)
        # The coordinates are then turned into a Pandas DataFrame
        coords = pd.read_csv(data_file_path, delimiter=";")

        if coords.empty:
            # Handle the case where there are no coordinates
            return render_template("csv_calc.html", iframe=None)

        # If the coordinates are not in the correct form, or adresses of String type have been uploaded, they are converted
        if coords['lat'].isnull().any() and coords['long'].isnull().any():
            # By using Nominatim, we can turn String addresses into geographical coordinates
            locator = Nominatim(user_agent="TSP_application_thesis")

            # Iterate through the provided addresses and convert them into coordinates using the locater.geocode function
            for index, row in coords.iterrows():
                address = f"{row['number']} ,{row['street']}, {row['city']}, {row['country']}"
                location = locator.geocode(address)
                coords.at[index, 'lat'] = location.latitude
                coords.at[index, 'long'] = location.longitude

            # Check if the provided coordinate sare in Float data type. If yes, they are saved in two lists
            if coords["lat"].dtype == float and coords["long"].dtype == float:
                lat = coords["lat"].tolist()
                long = coords["long"].tolist()

            # If they are of any other type, they are converted to a Float and turned into two lists
            else: 
                lat = coords["lat"].astype(float).tolist()
                long = coords["long"].astype(float).tolist()

        # If coordinates were provided and are in float type
        elif coords["lat"].dtype == float and coords["long"].dtype == float:
            lat = coords["lat"].tolist()
            long = coords["long"].tolist()

        # If coordinates were provided but are not in float type 
        else:
            lat = coords["lat"].astype(float).tolist()
            long = coords["long"].astype(float).tolist()

        # Round the lat, long columns to 6 decimal places for the optimal accuracy for later usage in folium and ors
        coords["lat"] = coords["lat"].round(6)
        coords["long"] = coords["long"].round(6)

        # Convert the DataFrame to JSON and update the corresponding session variable
        json_coords = coords.to_json()
        session["json_coords"] = json_coords

        # Initialize two lists for storing the coordinates in the correct format for folium and ors
        points = []
        # Iterate through very element of the lists and append them to the folium list
        for i in range(len(lat)):
            points.append([lat[i], long[i]])

        # Openrouteservice only takes in long, lat format, so a second list needs to be initialized
        points_ors = []
        for i in range(len(lat)):
            points_ors.append([long[i], lat[i]])

        # Obtaining the coordinate so fthe waypoints from the ors API
        response = client.directions(coordinates = points_ors, profile = "driving-car", format="geojson")
        route_coords = response["features"][0]["geometry"]["coordinates"]

        # Convert back to lat, long format fo rthe usage in folium
        route_coords = [[coord[1], coord[0]] for coord in route_coords]

        # Initialize a global variable for the marker group
        global marker_group
        # Use the FeatureGroup feature by folium, for storing and accessing multiple objects
        marker_group = folium.FeatureGroup(name = "CSV Data")
        # Create a marker for every coordinate in the coords variable
        for index, row in coords.iterrows():
            lat = row["lat"]
            long = row["long"]
            
            # Create a default marker
            marker = folium.Marker(location=[lat, long])
            # Add it to the marker group
            marker_group.add_child(marker)

        # Initialize a new map object, overwriting the default, empty map
        m = folium.Map(location=route_coords[0], tiles="cartodbpositron")
        # Set the max height to 600px for optimal rendering
        m.get_root().height = "600px"
        # Add the marker group to the map
        m.add_child(marker_group)
        # Render the map object
        iframe = m.get_root()._repr_html_()

        # Return the corresponding template as well as the upadted iframe for handling it in the HTML file
        return render_template("csv_calc.html", iframe=iframe, m=m)
    
    # If the priorly created boolean is still set to False, no data has been uploaded by the user, thus there is nothing to plot
    else:
        # In order to prevent an error, the map is initialized again with the same properties as the default map
        m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
        m.get_root().height = "600px"
        # Additionally a warning message is created, stating that data needs to be uploaded prior to plotting
        warning_message_plot = "Warning: No data uploaded yet."
        # Render the map object
        iframe = m.get_root()._repr_html_()

        # Return the corresponding template, the empty map and the warning message is passed to the HTML file 
        return render_template("csv_calc.html", iframe=iframe, warning_message_plot=warning_message_plot)

# View that runs the GA as well as the 2-opt algorithm, for plotting the route and the corresponding directions
@views.route("/distances-csv/", methods=["POST", "GET"])
def calculate_csv_distance():
    # Check if data has been uploaded, by accessig the sessions trigger boolean
    if session.get("file_uploaded") == True:

        # User default paramters suggested by the sko library at first
        max_iter = 200 # Maximum number of iterations the algorithm can perform
        size_pop = 50 # Population size
        prob_mut = 1 # Probability of mutation
        
        # Get the coordinates from the session 
        json_coords = session.get("json_coords")
        # Turn the coordinates back to a Pandas DataFrame
        coords = pd.read_json(json_coords)

        # Access the best paramters for the differently sized problems, which were earlier calculated using grid_search.py and are saved in a csv
        best_params_ga = pd.read_csv(r"C:\Users\Timmy Gerlach\Documents\Uni\Master\Masterarbeit\App\static\files\best_params_grid_search.csv", sep = ";")

        # Get the corresponding best values for the parameters according to the problem size. The problem size is give bei the length of the coordinates DataFrame
        if len(coords) in best_params_ga["problem_size"].values:
            row_index = best_params_ga.loc[best_params_ga["problem_size"] == len(coords)].index[0]
            max_iter = best_params_ga.loc[row_index, "max_iter"]
            size_pop = best_params_ga.loc[row_index, "pop_size"]
            prob_mut = best_params_ga.loc[row_index, "prob_mut"]

        # The problem size is given by the lengths of the coords data frame
        num_points = len(coords.index)

        # Coordinates in long, lat format for ors
        points_coordinate_ors = np.array(coords[["long","lat"]])

        # Add the starting point to the end of the list to make it circular
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

        # Calculation of the distance matrix with ors built in fucntion distance_matrix
        response = client.distance_matrix(locations=points_coordinate_ors.tolist(), metrics=["distance"], profile=profile)
        # Save the distance matrix in a numpy array
        distance_matrix = np.array(response["distances"])
        # Divide it by 1000 to obtain kilometre lengths
        distance_matrix_km = distance_matrix / 1000
        
        # Objective distance function given by sko. 
        def cal_total_distance(routine):
            # Exract the number of points from teh routine array
            num_points, = routine.shape

            # Initialize the toal distance to 0
            total_distance = 0

            # Loop through each consecutive pair of the routine given. 
            for i in range(num_points):
                # Calculate the distance between the current point and the next point in a circular way
                total_distance += distance_matrix_km[
                    routine[i % num_points], routine[(i + 1) % num_points]
                ]

            # Return the total distance
            return total_distance
        
        # 2-opt algorithm gyiven by py2opt. Takes in the cities index and the distance matrix and performs 2-opt swops for enhancing the result of the GA later
        def two_opt(cities_names, dist_mat):
        # Create a RouteFinder object
            route_finder = RouteFinder(dist_mat, cities_names)
            # Find the best route using 2-opt algorithm
            best_distance, best_route = route_finder.solve()

            return best_route, best_distance

        # Run the Genetic Algorithm (GA). Takes in the objective function and all optimized parameters as well as the problem size, give by the lengths of the coords
        ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
        # Save the best order and the corresponding distance in two variables for later usage
        best_points, best_distance = ga_tsp.run()
        # Separate the best distance
        best_distance_ga = best_distance[0]
        # To the best points append the first point to get a circular tour
        best_points = np.concatenate([best_points, [best_points[0]]])

        # Rearrangement of the points_coordinate variables to make them store the best tours coordinates. Priorly only the indexes were given in the variables
        best_tour_ga_ors = points_coordinate_ors[np.argsort(best_points)]
        
        # Convert teh best coordinates to a list for the usage in ors
        list_ga_ors = best_tour_ga_ors.tolist()

        # The 2-opt algorithms takes in city names so the index of the row is takes as "name"
        cities_names = [str(i + 1) for i in range(len(list_ga_ors))]

        # Applying 2-opt optimization
        optimized_route, optimized_distance = two_opt(cities_names, distance_matrix_km)

        # Rearrange the coordinates according to 2-opt. New_order only contains a newly ordered list of the indices of the coordinates not their real value
        new_order = [int(city) - 1 for city in optimized_route]

        # Get the real values for the coordinates
        updated_list_ga_ors = [list_ga_ors[index] for index in new_order]

        # Again, append the first place to make the tour circular, because the 2-opt sometimes returnes error otherwise
        first_place = list_ga_ors[0]
        updated_list_ga_ors = list_ga_ors + [first_place] 
        # Store the final coordinates in the session
        session["final_coordinates"] = updated_list_ga_ors

        # Plotting the best route calculated by the GA
        response_ga = client.directions(coordinates = updated_list_ga_ors, profile = profile, format="geojson")
        
        # Get the coordinates for the route from ors
        route_coords_ga = response_ga["features"][0]["geometry"]["coordinates"]
        # Change the format for the usage in folium
        route_coords_ga = [[coord[1], coord[0]] for coord in route_coords_ga]
        
        # Get the directions of one segment, e.g. A to B
        waypoints = list(dict.fromkeys(reduce(operator.concat, list(map(lambda step: step["way_points"], response_ga["features"][0]["properties"]["segments"][0]["steps"] )))))
        directions_ga = folium.PolyLine(locations=[list(reversed(response_ga["features"][0]["geometry"]["coordinates"][index])) for index in waypoints], color = "green")
        # Store the best route in a folium.Polyline oject
        best_ga_route = folium.PolyLine(locations=route_coords_ga, color="red")

        # Save the route segments in a variable. E.g. segment A to B, B to C, etc. 
        route_segments = response_ga["features"][0]["properties"]["segments"]

        # Extract waypoints and instructions from all segments and steps using ors
        # Initialize two lists for storing the values
        all_waypoints = []
        all_instructions = []

        # Loop thorugh every step in all segments and get the cooresponding coordiantes of every way point as well as the concrete instructions
        for segment in route_segments:
            for step in segment["steps"]:
                all_waypoints.extend(step["way_points"])
                all_instructions.append(step["instruction"])

        # Initialize map object and zoom to the locations in the coordinates
        route_coords = [[coord[1], coord[0]] for coord in updated_list_ga_ors]
        m = folium.Map(location=route_coords[0], tiles="cartodbpositron")
        m.get_root().height = "600px"

        # Initialize a counter for the total steps across all segments, to be aple to plot the correct directions
        total_step_count = 1

        # Since many waypoint were overlapping each other, a offeset factor has to be implemented to change the coordinates by a small valueof 0.000025
        offset_factor = 0.000025
        coord_offset_map = {}

        # Iterate through each segment in the route_segments list
        for segment in route_segments:
            
            # Iterate through every step in the current segment
            for step_index, step in enumerate(segment["steps"]):
                lat, lon = reversed(response_ga["features"][0]["geometry"]["coordinates"][step["way_points"][0]])
                
                # Check if the coordinate has been encountered before
                if (lat, lon) in coord_offset_map:
                    # If yes, add the offset factor so the markers won't overlap
                    coord_offset_map[(lat, lon)] += offset_factor
                else:
                    # If no, set an initial offset
                    coord_offset_map[(lat, lon)] = offset_factor
                
                # Apply the offset to the latitude and longitude to avoid overlapping markers
                lat_with_offset = lat + coord_offset_map[(lat, lon)]
                lon_with_offset = lon + coord_offset_map[(lat, lon)]
                
                if total_step_count == 1:  # For the step enumerated with 1
                    # Add a custom green marker for the starting point of the route
                    pushpin = folium.features.CustomIcon(r'static/pictures/waypoint_marker_green.png', icon_size=(30, 30))
                else:
                    # Add custom blue markers for the other direction way points
                    pushpin = folium.features.CustomIcon(r'static/pictures/waypoint_marker.png', icon_size=(15, 15))
                
                # Extract teh instructions form the current step
                instruction = step["instruction"]

                # Get the correct coordinates with offset for the instruction markers
                marker_coords = [lat_with_offset, lon_with_offset]

                # When clicking on the waypoints, display their coordinates
                popup_text = f"Coordinates: {lat_with_offset}, {lon_with_offset}"

                # When hovering over it, display the next step with the corresponding number 
                tool_tip = f"Step {total_step_count}: {instruction}"

                # Initialize the markers for every instruction, with the corresponding pushpin and the tooltip 
                folium.Marker(location=marker_coords, icon=pushpin, popup=popup_text, tooltip=tool_tip).add_to(m)
                
                # Increase the step count to be able to display the correct number of steps when hovering over the marker
                total_step_count += 1  # Increment the total step count for the next step
        
        # Add the best route as a Polyline to the map 
        m.add_child(best_ga_route)

        # Add the marker group to the map
        m.add_child(marker_group)

        # Render the iframe
        iframe = m.get_root()._repr_html_()

        # Return the corresponding template, the iframe, the parameters and the best distance. 
        return render_template("csv_calc.html", iframe=iframe, max_iter=max_iter, num_points=num_points, size_pop=size_pop, prob_mut=prob_mut, best_distance_ga=best_distance_ga )

    # If no data has been uploaded
    else:
        # Initialize the default map 
        m = folium.Map(location=[47.4244818, 9.3767173], tiles="cartodbpositron")
        # Set the properties right
        m.get_root().height = "600px"
        # Render it
        iframe = m.get_root()._repr_html_()
        # Create a warning message, that data needs to be uploaded prior to calculating something
        warning_message_calculator = "Please upload data"

        # Return the corresponding template, the iframe and the warning message, which can then be used by the HTMl files
        return render_template("csv_calc.html", iframe=iframe, warning_message_calculator=warning_message_calculator)

# View for retrieving the corresponding data as a gpx file for further implementation in tools and devices
@views.route('/download-gpx/', methods=['GET'])
def download_gpx():
    # Get the final coordinates and the profile used in the session
    final_coordinates = session.get('final_coordinates')
    profile = session.get("profile")

    # Define the headers fo rthe API request according to the ors API documentation
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': API_KEY,  # Replace with your actual API key
        'Content-Type': 'application/json; charset=utf-8'
    }

    # Create the request body according to the ors API documentation 
    body = {"coordinates": final_coordinates}

    # Select the correct profile for the gpx request according to the user's currently used profile
    if profile == "driving-car":
        call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car/gpx', json=body, headers=headers)
    elif profile == "foot-walking":
        call = requests.post('https://api.openrouteservice.org/v2/directions/foot-walking/gpx', json=body, headers=headers)
    elif profile == "cycling-regular":
        call = requests.post('https://api.openrouteservice.org/v2/directions/cycling-regular/gpx', json=body, headers=headers)
    else:
        call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car/gpx', json=body, headers=headers)
    
    # Call status 200 refers to a successful response, but does not automatically return a gpx file, which is why this step is included
    if call.status_code == 200:
        gpx_data = call.text

        # Initialize the gpx file.
        headers = {
            'Content-Disposition': 'attachment; filename="route.gpx"',
            'Content-Type': 'application/gpx+xml'
        }
        # Return the gpx file as a Response
        return Response(gpx_data, headers=headers)
    else:
        # Otherwise return an error
        return "Error: Unable to retrieve GPX data"

#NOT ACCESSABLE FOR THE USER BY DEFAULT #########################################################################################################
#section for the comparison of the algorithms. Use the route /calculator in the app for accessing it.
#The code for creating the plots etc. was taken from the SKO documentation website
@views.route("/calculator/", methods=["GET", "POST"])
def calculate_aco():

    # Get the user inout values for the parameters 
    if request.method == "POST":
        num_points = int(request.form["num_points"])
        max_iter = int(request.form["max_iter"])
        size_pop = int(request.form["size_pop"])
        prob_mut = int(request.form["prob_mut"])
    # If nothing has been entered by the user, default values are taken for the calculations of the algorithms
    else:
        num_points = 10
        max_iter = 200
        size_pop = 50
        prob_mut = 1

    # Generate random points and calculate a distance matrix
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    # Objective function
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    
    # Ant colony optimization (ACO)
    # Set a start time for the time measurement
    start_time_aco = time.time()

    # Initialize the ACO element, taking in the parameters an teh distance matrix
    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
               size_pop = size_pop, max_iter=max_iter,
              distance_matrix=distance_matrix)

    # Run the ACO 
    best_x, best_y = aca.run()

    # End the time measurement
    end_time_aco = time.time()

    # Calculate the total time taken by the ACO
    total_time_aco = end_time_aco - start_time_aco

    # Plot the problem as well as the improvements with each iteration made by the ACO
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.title("ACO Output & Performance")
    plt.savefig("static/pictures/aco.png")
    best_distance_aco = best_y
    
    # Genetic Algorithm (GA)
    # Set a start time for the time measurement 
    start_time_ga =time.time()

    # Initialize the GA element, taking in the parameters an teh distance matrix
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop = size_pop, max_iter = max_iter, prob_mut = prob_mut)
    
    # Run the GA
    best_points, best_distance = ga_tsp.run()

    # End the time measurement
    end_time_ga= time.time()

    # Calculate the total time taken by the GA
    total_time_ga= end_time_ga - start_time_ga

    # Plot the problem as well as the improvements with each iteration made by the GA
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    ax[1].plot(ga_tsp.generation_best_Y)
    plt.title("GA Output & Performance")
    plt.savefig("static/pictures/ga.png")
    best_distance_ga = best_distance[0]

    # Simulated annealing 
    start_time_sa = time.time()

    # Initialize the SA element, taking in the parameters an teh distance matrix
    sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)

    # Run the SA
    best_points, best_distance = sa_tsp.run()
    print(best_points, best_distance, cal_total_distance(best_points))

    end_time_sa = time.time()
    total_time_sa = end_time_sa - start_time_sa

    # Plot the problem as well as the improvements with each iteration made by the SA
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

    # Return frontend and variables
    return render_template("calculator.html", num_points=num_points, max_iter=max_iter, size_pop=size_pop, prob_mut=prob_mut, plot_url_aco="static/pictures/aco.png", plot_url_ga="static/pictures/ga.png", plot_url_sa="static/pictures/sa.png", total_time_aco=total_time_aco, total_time_ga=total_time_ga, total_time_sa=total_time_sa, best_distance_aco=best_distance_aco, best_distance_ga=best_distance_ga, best_distance_sa=best_distance_sa)
