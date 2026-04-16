# backend/modules/db.py

import mysql.connector


def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # XAMPP default boş olur genelde
        database="seng384"
    )
    return connection