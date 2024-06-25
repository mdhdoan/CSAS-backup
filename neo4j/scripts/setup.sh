#!/bin/bash

if [ -d "deployment/neo4j/data" ]; then
    echo "Neo4j is already setup ❌";
    exit 1
fi

./scripts/create_volumes.sh deployment/neo4j deployment/neo4j/data deployment/neo4j/import deployment/neo4j/logs deployment/neo4j/plugins
./scripts/create_volumes.sh deployment/ollama deployment/ollama/ollama deployment/live-qa/huggingface deployment/live-qa/pip

if [[ $(uname -s) == 'Linux' ]]; then
    ./neo4j/scripts/prepare-dotenv.sh "5.20.0" "5.20.0" "5.20.0" "2.6.7" "2.4.8" neo4j "phac@2024" 8G 16G CSAS
elif [[ $(uname -s) == "Darwin" ]]; then
    ./neo4j/scripts/prepare-dotenv.sh "5.20.0" "5.20.0" "5.20.0" "2.6.7" "2.3.5" neo4j "phac@2024" 8G 16G CSAS
else
    echo Unsupported platform $(uname -s). Only Linux or MacOS are supported.
    exit 1
fi

./neo4j/scripts/neo4j-download-plugins.sh