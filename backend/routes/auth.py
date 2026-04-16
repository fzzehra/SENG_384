from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash

from backend.modules.db import db
from backend.modules.models import User


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

        existing_user = User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first()
        if existing_user:
            flash("A user with that email or username already exists.", "error")
            return render_template("register.html", **_template_context())

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session["username"] = user.username
        flash("Registration successful. You are now logged in.", "success")
        return redirect(url_for("home"))

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

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid credentials. Please try again.", "error")
            return render_template("login.html", **_template_context())

        session["user_id"] = user.id
        session["username"] = user.username
        flash("Logged in successfully.", "success")
        return redirect(url_for("home"))

    return render_template("login.html", **_template_context())


@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))
