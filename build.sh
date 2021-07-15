#!/bin/bash

COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build --progress plain

if [ ! -d engine/pretrained_models ]; then
    mkdir engine/pretrained_models
fi
