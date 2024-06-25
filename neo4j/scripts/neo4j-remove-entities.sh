#!/bin/bash

set -e

source deployment/neo4j/.env

echo 'Cleanup entities ...'
sudo cp neo4j/cql/remove-entities.cql deployment/neo4j/import/.
docker exec -u ${NEO4J_USERNAME} --interactive --tty  neo4j cypher-shell -u ${NEO4J_USERNAME} -p ${NEO4J_PASSWORD} --file /import/remove-entities.cql
echo 'Entities are removed ✅'
