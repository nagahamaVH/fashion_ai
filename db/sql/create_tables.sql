DROP TABLE IF EXISTS "public"."images";
CREATE TABLE "public"."images" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "height" int,
  "width" int,
  "train_test" text
);

DROP TABLE IF EXISTS "public"."segmentation";
CREATE TABLE "public"."segmentation" (
  "id" SERIAL PRIMARY KEY,
  "img_id" int REFERENCES images (id),
  "encoded_pixels" text,
  "class_id" int,
  "attribute_id" text
);

DROP TABLE IF EXISTS "public"."categories";
CREATE TABLE "public"."categories" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "supercategory" text,
  "level" int
);

DROP TABLE IF EXISTS "public"."attributes";
CREATE TABLE "public"."attributes" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "supercategory" text,
  "level" int
);