#!/bin/bash

docker-compose run -d db
docker-compose run -d adminer
docker-compose run engine
