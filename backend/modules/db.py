import os

import pymysql


def get_db_connection():
    ssl_ca = os.environ.get("DB_SSL_CA")
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "facial_app"),
        port=int(os.environ.get("DB_PORT", "3307")),
        ssl={"ca": ssl_ca} if ssl_ca else None,
        cursorclass=pymysql.cursors.DictCursor,
    )