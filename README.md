# Developing a Web Application for the Traveling Salesman Problem 

1. [About](#about)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Features (for Developers)](#features-for-developers)
   - [app.py](#apppy)
   - [views.py](#viewspy)
   - [style.css](#stylecss)
   - [Other Files](#other-files)
4. [Usage](#usage)
5. [Results](#results)
6. [Future developments and Contribution](#future-developments-and-contribution)
7. [Contact](#contact)


## About
In this project, a Genetic Algorithm is used to solve the TSP for small-sized real-life problems. It is therefore implemented in a Flask web application hosted with Pythonanyhwere.com. 
It can be accessed via https://vantablack9000.pythonanywhere.com. On the website you can upload a list of addresses or coordinates or manually set up your data base. The GA used in the code then works with dat retrieve from the open route service API.
This project has been part of a Master Thesis obtained at the University of St. Gallen. It was submitted to Prof. Dr. Ivo Blohm from the Institut für Wirtschaftsinformatik (IWI-HSG).
## Getting Started

Follow these steps to get started with the Genetic Algorithm-based Traveling Salesperson Problem Solver:

### Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Python (>=3.6)
- Flask (Install using `pip install Flask`)
- numpy (Install using `pip install numpy`)
- scipy (Install using `pip install scipy`)
- matplotlib (Install using `pip install matplotlib`)
- sko (Install using `pip install scikit-optimize`)
- folium (Install using `pip install folium`)
- openrouteservice (Install using `pip install openrouteservice`)
- geopy (Install using `pip install geopy`)
- geopandas (Install using `pip install geopandas`)
- gpxpy (Install using `pip install gpxpy`)
- py2opt (Install using `pip install py2opt`)
- tqdm (Install using `pip install tqdm`)
- flask_caching (Install using `pip install Flask-Caching`)

### Installation

1. Clone the repository to your local machine using Git:

   ```bash
   git clone https://github.com/VantaBlack9000/App.git


2. Activate your virtual environment if you have one (recommended)

4. Install requirements.txt

   ```bash
   pip install -r requirements.txt

5. Change directory to the main folder /App

   ```bash
   cd directory/APP

6. Get a ors API key and replace it with the placeholder in views.py (https://openrouteservice.org/dev/#/api-docs)

7. Ready to use!

## Features (for Developers)
The following section explains in details the different files as well as their most important functionalities

### app.py
app.py stores the applications main source code. By running this file, you run the main Flask application on your localhost. It accesses the views created in views.py, so make sure to run views.py before tryng to run the application. Once you executed app.py, depending on your IDE, in your console the correct path for accessing the application shoul pop up. I recommend using Visual Studio Code for projects like this, since it offers a clear overview of the projects structure. Nevertheless, by default you should also be able to access the applications frontend after running views.py and app.py by tying in localhost:8000 to your browser of choice. 

### views.py
views.py is bar far the most important file of the project since it runs all the calculations and main functionalities of the application. It consists of multiple sections that will be explained in the following:

#### Section for importing the libraries and packages used for this project
Right at the top of the section you will see the necessary packages and libraries. Once you installed the requirements.txt file and a version of Python on your local device, no errors should be encountered when it comes to the packages. Nevertheless, depending on the time of your access, there might be newer versions of certain packages available, than the ones listed in requirements.txt, which is why I advice you to double-check this. 

#### Important initial variables
The next section defines some variables such as the the folium map used throughout the following Flask routes and functions, that had to be initialized globally so they are accessable for usage in the whole file. Also, in this sectio, the API key is stored in a variable. Please replace the placeholder with your API key retrieved from ors. Next, the views variable is initialized, which is crucial for the import to app.py and for ultimately being able to run the whole application. No actions needed in this section, except for replacing the API key. 

#### Modules section
In this section a single function called 
```python
remove_files_in_folder()
```
This module has been initialized outside of the routes, since it is used in multiple functions throughout the whole project in order to keep the upload folder clean. Replace the folder path with your actual path. 

#### @views.route("/")
The first route of the Flask application is the main landing. It consists of a function called
```python
home()
```
When called, this function clears the user session in order to grant a clean process and empties the upload folder from old files that might have been stored there. Finally, it returns the template 
```HTML
home.html
```
This template is the main landing of the application. 

#### @views.route("/about/")
The second route consists of a function called 
```python
about()
```
It works pretty much like the function defined in the first route but when executed returns the template
```HTML
about.html
```
This template is important to display the about page, on which my project is descibed to the end users, several disclaimers are maide and where the main file of my thesis will be available. 

#### @views.route("/csv-calculator/")
The third route consists of a function called 
```python
upload_csv()
```
As the name of the function indicates, this function, when executed via a button in the corresponding .html file, allows the user to upload a .csv file consisting of either adresses or coordinates for the further usage in the application. When executed, this function returns the template 
```HTML
csv_calc.html
```
Also, this function returns an iframe, a HTML representation of the map object that has been initialiued at the beginning of the views.py file. If no Csv file has been uploaded it also returns a warning message to the user. The HTML template returned by the function is the most important landing of the application as it allows the user to calculate tours by offering the corresponding user interface for using the calculator. Please note, that the detailed functioning of the methods used in this route is commented out. So, for a deeper understanding make sure to read the corresponding chapters in the thesis as well as the comments along the code. 

#### @views.route("/csv-calculator-data/")
After successfully uploading his data, the user is able to review his data by viewing it through a function defined in this route. It is called
```python
show_data()
```
This function first checks if a uploaded file is existant and afterwards either returns the same HTML template as the prior route but with an additional HTML representation of a Pandas Dataframe in tabular form. If no has been uploaded, but the function is executed nevetheless, it returns the template and a warning message indicating that the user must uplaod a file before being able to check his data. Again, the detailed line by line instructions can be retrieed from the corresponding thesis chapters and the comments in the code. 

#### @views.route("/add-customers/")
Now the user successfully uploaded data and reviewed it, he can add customers manually to the data if he likes to. Also, if he decides against uploading a file with adresses or coordinates, he can manually input his database using the formfield provided in the template. This routes main function is called
```python
manually_add_customers()
```
Its main function is to convert addresses into geo coordinates and to append them to the existing data or to set up a new database if no data exists up until now. The function again renders the same template as before.

#### @views.route("/plotted-data/")
Now the user is all set, and his data has correctly been uplaoded, he can plot the places on the folium map by using a function called
```python
plot_csv()
```
This function adds folium markers to every location the user uploaded or typed in manually by accessing the ors API. Also, if a file full of addresses has been uploaded, it converts them to coordinates just as the function described above. 

#### @views.route("/distances-csv/")
The most important route of the views.py file is this one. It inherits the Genetic Algorithm this application is centered around. It takes in the user data, calculates the shortes route between all points and returnes them on the map with the help of a folium Polyline. The algorithm takes in parameters that were priorly found to be best by a grid search algorithm. The results obtained form the GA are then being tuned by a 2-opt algorithm provided by the py2opt library. By making use of the ors API, it calculates the actual drivable distances using actual roads. It also adds direction instructions to every important point the users way and adds them on folium customized markers. The main function of the route is called
```python
calculate_csv_distance()
```

#### @views.route('/download-gpx/')
This route inherits a function called
```python
download_gpx()
```
This function gain calls the ors APi and converts the data calculated in the above described route to GPX data so the user can download and use the data on his navigation devices or application such as GoogleMaps, TomTom or Garmin. 

#### @views.route("/calculator/")
Lastly, this route is especially of interested if you're interested in comparing a Genetic Algorithm to Ant Colony Optimization and Simulated Annealing. This route can only be accessed by tying in the route directly behind the main applications url. If you run the application on your local device you can access it using localhost:8000/calculator/. The main function of this route allows you to input different parameters for the algorithms. It will thn run calculations on all three so you will be able to see performance differences. It was mainly used to decide for a suitable algorithm. 

### HTML files
All in all there are four HTML files in /templates directory of this repository:
```HTML
about.html, calculator.html, csv_calc.html and home.html
```
These files are written in HTML and contain the main content of the application. They are commented out accordingly.

### style.css
The style file of the application can be found in /static/css/style.css. It has been used to enhace the user experience and the user interface. Comments are added accordingly. 

### Other Files
other_files contains two files, namely grid_search.py in which the best parameters fo rthe GA to take in have been determined and exact_vs_ga.py in which the GA has been compared performance wise to a Dynamic Programming Algorithm. 

## Usage
This section explains how to use the application to retrieve routes

1. Visit vantablack9000.pythonanywhere.com and read through the instructions of the appliction. Navigate using the small black arrows at the bottom of each explanatory section.
2. Retrieve a sample file at the bottom section of the home page by clicking the download button. Afterwards press the get started button. 

![image](https://github.com/VantaBlack9000/App/assets/92924370/e0c9087e-9e66-473c-ba47-7e2a69428bab)

3. Upload a csv file using the upload module on the age you were directed to

    ![image](https://github.com/VantaBlack9000/App/assets/92924370/e40dd070-13f9-4e00-a260-9ed9b3a6b733)
  
    OR
  
    Manually input the adresses you want to visit using the form fields displayed below
    ![image](https://github.com/VantaBlack9000/App/assets/92924370/cc32fc98-2886-466e-8aba-8c849a381f7f)

4. Use the show CSV button to review your data. It should look something like this:
  ![image](https://github.com/VantaBlack9000/App/assets/92924370/076e5f56-2651-40cf-a565-83f55000140f)

5. Plot your data using the plot data button at the bottom section:
   ![image](https://github.com/VantaBlack9000/App/assets/92924370/8ed154c8-481f-4ff4-8900-e1abccd26e44)

   The result should be your data plotted on the map like this:
   ![image](https://github.com/VantaBlack9000/App/assets/92924370/421f6101-ecae-4d6c-bb25-13f152c7cc27)

6. Select your vehicle by using the drop down above and press the calculate button to see the shortest route calculated by the algorithm. The result should look like this:
   ![image](https://github.com/VantaBlack9000/App/assets/92924370/c8ad4c90-5596-40a5-a952-76802cf29c47)

   The green marker is the recommended starting location. Every small blue marker along the red line displays the according step number and instruction when hovering over it and its coordinates when clicking on it (Desktop). When using the application on your mobile devices you will need to tap on the small marker to be able to see the instructions and the coordinates.

7. Retrieve your route as a detailed gpx file by clicking on the download botton at the bottom of the page
  ![image](https://github.com/VantaBlack9000/App/assets/92924370/709e4aea-4ac8-402a-83cb-76d7d88f47b3)

8. Insert the file into one of the tools or devices listed at the bottom of the page. Detailed instructions of how to use the file in one of these can be retrieved when clicking on teh corresponding buttons.
  ![image](https://github.com/VantaBlack9000/App/assets/92924370/1541b1a2-0f70-409b-bb9d-9b06b7b183be)

9. All done! Create new routes or play around with the tool as you like. 

## Results
This application is mainly set up for scientific research purposes. I do not claim that it works perfectly nor that it finds the perfect route all the time. When testing the algorithm against an exact method on problems of size 10-20, the average deviation from the best route obtained by the GA was 7.23%. This might sound like a minor deviation but can affect the efficiency of the routes. So, ulitimately I cannot recommend to use this software in a professional context. If you are interested in the underlying problem and its solution approaches, the tool might still be of interest for you, since it offers interesting insights. I also recommend to read my thesis which will be available here once it has been reviewed if you are interested in getting detailed instructions on how to develop a project like this. Also, for inquiries of all kinds, feel free to contact me via timmy.gerlach@student.unisg.ch at any time!

## Future developments and Contribution
There could be many interesting possibilities to further improve the applcation. 

- Using a differnet underlying algorihtm: There any many solving methods for small sized TSP. A detailed list can be seen in my thesis. To implement different algorithms could be a useful addition to the research and the performance of the application
- Solving a different TSP version: In my thesis you will also find an overview of existing variants of the classic TSP. Changing the applications main objective to solving one of these variants could be another useful addition to reseach and usability.

## Contact
Email: timmy.gerlach@student.unisg.ch
LinkedIn: https://www.linkedin.com/in/tim-florian-gerlach-193096150/?originalSubdomain=de

---

