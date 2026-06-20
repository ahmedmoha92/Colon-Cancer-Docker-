#!/bin/bash
# run.sh — Commandes de gestion du pipeline Colon Cancer
# Usage: bash run.sh [commande]
#
# Commandes disponibles:
#   up       — Build + lancer tout (train + API)
#   up-d     — Build + lancer en arrière-plan
#   down     — Stopper les conteneurs
#   clean    — Stopper + supprimer le volume (repart de zéro)
#   train    — Re-entraîner uniquement le modèle
#   logs     — Voir les logs de l'API en temps réel
#   test     — Tester tous les endpoints de l'API
#   status   — Afficher le statut des conteneurs
#   docs     — Afficher les URLs de documentation

set -e

CMD=${1:-help}

case "$CMD" in
  up)
    echo "🚀 Lancement du pipeline complet (build + train + API)..."
    docker compose up --build
    ;;
  up-d)
    echo "🚀 Lancement en arrière-plan..."
    docker compose up --build -d
    echo "✅ API disponible sur http://localhost:8000"
    echo "   Logs : bash run.sh logs"
    ;;
  down)
    echo "⏹  Arrêt des conteneurs..."
    docker compose down
    ;;
  clean)
    echo "🧹 Nettoyage complet (conteneurs + volume)..."
    docker compose down -v
    docker rmi -f colon_cancer_project-ml_service colon_cancer_project-api_service 2>/dev/null || true
    echo "✅ Nettoyage terminé"
    ;;
  train)
    echo "🧬 Re-entraînement du modèle..."
    docker compose run --rm ml_service
    ;;
  logs)
    docker compose logs -f api_service
    ;;
  test)
    echo "🧪 Tests de l'API..."
    python3 test_api.py --host http://localhost:8000
    ;;
  status)
    docker compose ps
    ;;
  docs)
    echo "📄 Swagger UI : http://localhost:8000/docs"
    echo "📄 ReDoc      : http://localhost:8000/redoc"
    ;;
  help|*)
    echo ""
    echo "Usage: bash run.sh [commande]"
    echo ""
    echo "  up       Build + lancer tout (train + API)"
    echo "  up-d     Build + lancer en arrière-plan"
    echo "  down     Stopper les conteneurs"
    echo "  clean    Stopper + supprimer volume et images"
    echo "  train    Re-entraîner uniquement le modèle"
    echo "  logs     Logs de l'API en temps réel"
    echo "  test     Tester tous les endpoints"
    echo "  status   Statut des conteneurs"
    echo "  docs     URLs de la documentation"
    echo ""
    ;;
esac
