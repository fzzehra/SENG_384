import os
import time

from flask import Flask, render_template, session, redirect, url_for, flash
from flask_cors import CORS

from backend.modules.utils.helpers import ensure_dir
from backend.routes.upload import upload_bp
from backend.routes.process import process_bp
from backend.routes.transform import transform_bp
from backend.routes.analysis import analysis_bp
from backend.routes.metrics import metrics_bp
from backend.routes.export import export_bp
from backend.routes.auth import auth_bp


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
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["RESULT_FOLDER"])

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
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("auth.login"))
        return render_template("upload.html")

    @app.route("/controls-page")
    def controls_page():
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("auth.login"))
        return render_template("controls.html")

    # 🔥 CACHE FIX BURADA
    @app.route("/preview-page")
    def preview_page():
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("auth.login"))
        return render_template("preview.html", cache_buster=int(time.time()))

    @app.route("/result-page")
    def result_page():
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("auth.login"))
        return render_template("result.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)