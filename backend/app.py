import os
"""
Launching Flask
Adding route files to the system
Creating upload and results folders
Returning a test message on the main page
"""

from flask import Flask, render_template, session
from flask_cors import CORS
from sqlalchemy.engine import make_url

from backend.modules.db import db
from backend.modules.utils.helpers import ensure_dir
from backend.routes.upload import upload_bp
from backend.routes.process import process_bp
from backend.routes.transform import transform_bp
from backend.routes.analysis import analysis_bp
from backend.routes.metrics import metrics_bp
from backend.routes.export import export_bp
from backend.routes.auth import auth_bp


def create_database_if_not_exists(sqlalchemy_uri: str) -> None:
    url = make_url(sqlalchemy_uri)
    if not url.drivername.startswith("mysql"):
        return

    import pymysql

    server_connection = pymysql.connect(
        host=url.host or "localhost",
        user=url.username or "root",
        password=url.password or "",
        port=url.port or 3306,
        charset="utf8mb4",
    )
    try:
        with server_connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{url.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        server_connection.commit()
    finally:
        server_connection.close()


def create_app():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static")
    )
    CORS(app)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret")
    app.config["UPLOAD_FOLDER"] = "static/uploads"
    app.config["RESULT_FOLDER"] = "static/results"
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@localhost/seng384"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["RESULT_FOLDER"])

    create_database_if_not_exists(app.config["SQLALCHEMY_DATABASE_URI"])
    db.init_app(app)

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(process_bp, url_prefix="/process")
    app.register_blueprint(transform_bp, url_prefix="/transform")
    app.register_blueprint(analysis_bp, url_prefix="/analyze")
    app.register_blueprint(metrics_bp, url_prefix="/metrics")
    app.register_blueprint(export_bp, url_prefix="/export")

    @app.context_processor
    def inject_current_user():
        return {"current_user": session.get("username")}

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

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)