from flask import Flask
from flask_caching import Cache
from views import views
import os 

#setting up upload folder and adding it to flask
UPLOAD_FOLDER = os.path.join('static', 'map_csv_files')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

app.register_blueprint(views, url_prefix="/")
app.use_static_for_assets = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "my secret key for my thesis"
app.debug = True

if __name__ == "__main__":
    app.run(debug=True, port=8000)

