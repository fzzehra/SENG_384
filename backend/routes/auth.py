import traceback
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
from backend.modules.db import get_db_connection

auth_bp = Blueprint("auth", __name__)


def _template_context():
    return {"current_user": session.get("username")}


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("Please complete all required fields.", "error")
            return render_template("register.html", **_template_context())

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html", **_template_context())

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE email = %s OR username = %s",
                (email, username)
            )
            existing_user = cursor.fetchone()

            if existing_user:
                cursor.close()
                conn.close()
                flash("A user with that email or username already exists.", "error")
                return render_template("register.html", **_template_context())

            hashed_password = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            conn.commit()

            cursor.close()
            conn.close()

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("auth.login"))

        except Exception as e:
            print("REGISTER ERROR TYPE:", type(e))
            print("REGISTER ERROR REPR:", repr(e))
            traceback.print_exc()
            flash("Registration failed. Check terminal.", "error")

    return render_template("register.html", **_template_context())


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "error")
            return render_template("login.html", **_template_context())

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            if not user:
                flash("Invalid credentials. Please try again.", "error")
                return render_template("login.html", **_template_context())

            if not check_password_hash(user["password"], password):
                flash("Invalid credentials. Please try again.", "error")
                return render_template("login.html", **_template_context())

            session["user_id"] = user["id"]
            session["username"] = user["username"]

            flash("Logged in successfully.", "success")
            return redirect(url_for("home"))

        except Exception as e:
            print("LOGIN ERROR TYPE:", type(e))
            print("LOGIN ERROR REPR:", repr(e))
            traceback.print_exc()
            flash("Login failed. Check terminal.", "error")

    return render_template("login.html", **_template_context())


@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("You have been logged out.")
    return redirect(url_for("auth.login"))