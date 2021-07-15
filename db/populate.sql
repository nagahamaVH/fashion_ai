CREATE EXTENSION pgcrypto;

INSERT INTO "user" (name, login, password) VALUES (
  'Admin',
  'admin',
  crypt('teste123', gen_salt('bf'))
);
