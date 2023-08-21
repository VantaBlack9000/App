# Import necessary libraries and packages
from flask import Flask
from flask_caching import Cache
from views import views
import os 

# Setting up upload folder and adding it to flask
UPLOAD_FOLDER = os.path.join('static', 'map_csv_files')
ALLOWED_EXTENSIONS = {'csv'}

# Create a Flask App instance
app = Flask(__name__)

# Register the 'views' blueprint with the app and set its URL prefix
app.register_blueprint(views, url_prefix="/")

# Allow the app to serve static files directly from the 'static' folder
app.use_static_for_assets = True

# Configure the app's settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "my secret key for my thesis"

# Main application entry point
if __name__ == "__main__":
    app.run(debug=True, port=8000)

