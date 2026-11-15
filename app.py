"""The web layer. It only moves data between the website and minilearn.py.
All the machine learning lives in minilearn.py."""
import os
import tempfile

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import minilearn as ml
import explore as ex

app = Flask(__name__, static_folder="static")
MAX_ROWS = 2000  # larger files are sampled, so training stays quick


def read_upload(upload):
    """minilearn reads from a file path, so save the upload to a temporary file first."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(upload.read())
        path = tmp.name
    try:
        return ml.load_csv(path)
    finally:
        os.remove(path)


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        header, rows = read_upload(upload)
    except Exception:
        return jsonify(error="That file could not be read as a CSV."), 400
    if len(header) < 2 or len(rows) == 0:
        return jsonify(error="The file needs a header row, at least two columns and some data rows."), 400

    columns = ml.profile_columns(header, rows)

    # Step 1: no target chosen yet, so only describe the data
    target = request.form.get("target")
    if not target:
        return jsonify(rows=len(rows), columns=columns)

    # Step 2: train and rank
    if target not in header:
        return jsonify(error="That column is not in the file."), 400
    tune = request.form.get("tune", "1") == "1"
    task = ml.detect_task(columns, target)

    sampled = len(rows) > MAX_ROWS
    if sampled:
        keep = np.random.default_rng(0).choice(len(rows), MAX_ROWS, replace=False)
        rows = [rows[i] for i in keep]

    try:
        X, y, labels, names = ml.prepare_data(header, rows, target, task)
        if len(names) == 0:
            return jsonify(error="No usable feature columns were found."), 400
        if len(y) < 30:
            return jsonify(error=f"Only {len(y)} rows have a target value. At least 30 are needed for a fair comparison."), 400
        if task == "classification" and len(set(y)) < 2:
            return jsonify(error="The target column has only one distinct value, so there is nothing to predict."), 400
        report = ml.rank_models(X, y, task, names, tune)
    except Exception as err:
        return jsonify(error=f"Could not train on this file: {err}"), 400

    report.update(columns=columns, classes=labels, features=names, sampled=sampled)
    return jsonify(report)


@app.route("/api/explore", methods=["POST"])
def explore_route():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        header, rows = read_upload(upload)
        info = ml.profile_columns(header, rows)
        mode = request.form.get("mode")
        a, b = request.form.get("a"), request.form.get("b")
        if mode == "overview":
            return jsonify(ex.overview(header, rows, info))
        if mode == "compare":
            if a not in header or b not in header:
                return jsonify(error="Pick two columns from the file."), 400
            return jsonify(ex.compare(header, rows, info, a, b))
        if mode == "groups":
            k = max(2, min(6, int(request.form.get("k", 3))))
            return jsonify(ex.groups(header, rows, info, k))
        return jsonify(error="Unknown goal."), 400
    except ValueError as err:
        return jsonify(error=str(err)), 400
    except Exception as err:
        return jsonify(error=f"Could not study this file: {err}"), 400


if __name__ == "__main__":
    app.run(debug=True)