import os
import time

from flask import Flask, render_template, session, redirect, url_for, flash, request, jsonify, current_app
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
        return render_template("controls.html", cache_buster=int(time.time()))

    @app.route("/preview-page")
    def preview_page():
        if not session.get("user_id"):
            flash("Please log in to access this page.", "error")
            return redirect(url_for("auth.login"))
        return render_template("preview.html", cache_buster=int(time.time()))

    @app.route("/history")
    def history():
        if not session.get("user_id"):
            flash("Please log in to view your history.", "error")
            return redirect(url_for("auth.login"))

        try:
            from backend.modules.db import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_history WHERE user_id = %s ORDER BY created_at DESC",
                (session["user_id"],)
            )
            history_data = cursor.fetchall()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"FETCH HISTORY ERROR: {e}")
            history_data = []

        return render_template("history.html", history=history_data)

    @app.route("/hairstyles")
    def list_hairstyles():
        folder = os.path.join(app.static_folder, "hairstyles")
        if not os.path.exists(folder):
            return jsonify([])
        files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        return jsonify(files)

    @app.route("/save-history", methods=["POST"])
    def save_history():
        if not session.get("user_id"):
            return jsonify({"success": False, "message": "Not logged in"}), 401

        try:
            import shutil
            from backend.modules.db import get_db_connection

            data = request.get_json() or {}
            transform_type = data.get("transform_type", "custom")
            intensity = float(data.get("intensity", 1.0))

            static_folder = current_app.static_folder
            original_src = os.path.join(static_folder, "uploads", "original.jpg")
            transformed_src = os.path.join(static_folder, "uploads", "transformed.jpg")

            print(f"SAVE HISTORY: original exists={os.path.exists(original_src)}")
            print(f"SAVE HISTORY: transformed exists={os.path.exists(transformed_src)}")

            if not os.path.exists(original_src):
                return jsonify({"success": False, "message": "Original image not found"}), 400
            if not os.path.exists(transformed_src):
                shutil.copy(original_src, transformed_src)

            history_folder = os.path.join(static_folder, "results", "history")
            ensure_dir(history_folder)

            ts = int(time.time())
            orig_filename = f"orig_{ts}.jpg"
            result_filename = f"result_{ts}.jpg"

            shutil.copy(original_src, os.path.join(history_folder, orig_filename))
            shutil.copy(transformed_src, os.path.join(history_folder, result_filename))

            rel_original = f"static/results/history/{orig_filename}"
            rel_transformed = f"static/results/history/{result_filename}"

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_history (user_id, original_image, transformed_image, transform_type, intensity) VALUES (%s, %s, %s, %s, %s)",
                (session["user_id"], rel_original, rel_transformed, transform_type, intensity)
            )
            conn.commit()
            cursor.close()
            conn.close()

            print(f"SAVE HISTORY: success")
            return jsonify({"success": True})

        except Exception as e:
            import traceback
            print(f"SAVE HISTORY ERROR: {e}")
            traceback.print_exc()
            return jsonify({"success": False, "message": str(e)}), 500

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