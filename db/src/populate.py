import os
import csv
import psycopg2


db_name = os.environ["DB_NAME"]
db_user = os.environ["DB_USER"]
db_psw = os.environ["DB_PSW"]

conn = psycopg2.connect(
    "host=localhost dbname={} user={} password={} port=5431".format(
        db_name, db_user, db_psw))
cur = conn.cursor()

# Categories
with open('./data/categories_table.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        cur.execute(
            "INSERT INTO categories VALUES (%s, %s, %s, %s)", row)
conn.commit()

# Attributes
with open('./data/attributes_table.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        cur.execute(
            "INSERT INTO attributes VALUES (%s, %s, %s, %s)", row)
conn.commit()

# Images
with open('./data/images_table.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        cur.execute(
            "INSERT INTO images (name, height, width, train_test) VALUES (%s, %s, %s, %s)", row)
conn.commit()

cur.close()
conn.close()