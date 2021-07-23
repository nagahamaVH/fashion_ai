import os
import csv
import psycopg2

csv.field_size_limit(100000000)

db_name = os.environ["POSTGRES_DB"]
db_user = os.environ["POSTGRES_USER"]
db_psw = os.environ["POSTGRES_PASSWORD"]

conn = psycopg2.connect(
    "host=localhost dbname={} user={} password={}".format(
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
            "INSERT INTO images VALUES (%s, %s, %s, %s, %s)", row)
conn.commit()

# Segmentation
with open('./data/segmentation_table.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        cur.execute(
            "INSERT INTO segmentation (img_id, encoded_pixels, class_id, attribute_id) VALUES (%s, %s, %s, %s)", row)
conn.commit()

cur.close()
conn.close()
