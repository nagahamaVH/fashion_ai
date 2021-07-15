DROP TABLE IF EXISTS "public"."user";
CREATE TABLE "public"."user" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "login" text,
  "password" text,
  "created_at" timestamp DEFAULT (now())
);

DROP TABLE IF EXISTS "public"."images";
CREATE TABLE "public"."images" (
  "id" SERIAL PRIMARY KEY,
  "local" text,
  "height" int,
  "width" int,
  "group" text
);

DROP TABLE IF EXISTS "public"."segmentation";
CREATE TABLE "public"."segmentation" (
  "id" SERIAL PRIMARY KEY,
  "img_id" int,
  "encoded_pixels" text,
  "class_id" int,
  "attribute_id" int
);

DROP TABLE IF EXISTS "public"."categories";
CREATE TABLE "public"."categories" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "supercategory" int,
  "level" int
);

DROP TABLE IF EXISTS "public"."attributes";
CREATE TABLE "public"."attributes" (
  "id" SERIAL PRIMARY KEY,
  "name" text,
  "supercategory" int,
  "level" int
);
