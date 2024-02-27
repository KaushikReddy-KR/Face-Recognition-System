import pickle
from flask import Flask, render_template, request
from os.path import join, dirname
import numpy as np




app = Flask(__name__)


@app.route("/")
def hello_word():
    return {"ok" : "initialised"}


@app.route("/submit", methods=["POST"])
def recognise_face():
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)