#!/bin/bash

copy_data=$1

set -e

source deployment/neo4j/.env

echo 'Copying scripts and data ...'
sudo cp neo4j/cql/import-news-articles.cql deployment/neo4j/import/.
if [ "$copy_data" = "copy" ]; then
    sudo cp ../data/dfo/preprocessed/*.jsonl deployment/neo4j/import/.
    sudo cp ../data/dfo/clusters/*.jsonl deployment/neo4j/import/.
fi
echo 'Executing queries ...'
docker exec -u ${NEO4J_USERNAME} --interactive --tty  neo4j cypher-shell -u ${NEO4J_USERNAME} -p ${NEO4J_PASSWORD} --file /import/import-news-articles.cql
echo 'News article imported ✅'
