import os
"""
Launching Flask
Adding route files to the system
Creating upload and results folders
Returning a test message on the main page
"""

from flask import Flask, render_template
from flask_cors import CORS

from backend.routes.upload import upload_bp
from backend.routes.process import process_bp
from backend.routes.transform import transform_bp
from backend.routes.analysis import analysis_bp
from backend.routes.metrics import metrics_bp
from backend.routes.export import export_bp
from backend.modules.utils.helpers import ensure_dir


def create_app():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static")
    )
    CORS(app)

    app.config["UPLOAD_FOLDER"] = "static/uploads"
    app.config["RESULT_FOLDER"] = "static/results"
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["RESULT_FOLDER"])

    app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(process_bp, url_prefix="/process")
    app.register_blueprint(transform_bp, url_prefix="/transform")
    app.register_blueprint(analysis_bp, url_prefix="/analyze")
    app.register_blueprint(metrics_bp, url_prefix="/metrics")
    app.register_blueprint(export_bp, url_prefix="/export")

    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/upload-page")
    def upload_page():
        return render_template("upload.html")

    @app.route("/controls-page")
    def controls_page():
        return render_template("controls.html")

    @app.route("/preview-page")
    def preview_page():
        return render_template("preview.html")

    @app.route("/result-page")
    def result_page():
        return render_template("result.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)