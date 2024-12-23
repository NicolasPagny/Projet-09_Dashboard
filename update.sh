#!/bin/bash

# 1) Effectuer un git pull pour récupérer les dernières modifications
echo "Mise à jour du dépôt git..."
git pull origin main

# 2) Rebuilder et redémarrer les conteneurs avec Docker Compose
echo "Rebuild et démarrage des conteneurs Docker..."
docker compose up --build -d

echo "Mise à jour terminée avec succès!"