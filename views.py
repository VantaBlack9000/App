from flask import Blueprint, render_template

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

@views.route("/calculator/")
def calculator():
    return render_template("calculator.html")

@views.route("/contact/")
def contact():
    return render_template("contact.html")

@views.route("/sources/")
def sources():
    return render_template("sources.html")