# ══════════════════════════════════════════════════
#  Colon Cancer ML Pipeline — Makefile
#  Remplacer DOCKER_USER par ton username Docker Hub
# ══════════════════════════════════════════════════

DOCKER_USER = ahmedmoha
ML_IMAGE    = $(DOCKER_USER)/colon-ml-trainer
API_IMAGE   = $(DOCKER_USER)/colon-api
TAG         = latest

.PHONY: up up-d down clean train logs test status docs \
        build push push-ml push-api pull

# ── Local ──────────────────────────────────────────
up:
	docker compose up --build

up-d:
	docker compose up --build -d

down:
	docker compose down

clean:
	docker compose down -v
	docker rmi -f $(ML_IMAGE) $(API_IMAGE) 2>/dev/null || true

train:
	docker compose run --rm ml_service

logs:
	docker compose logs -f api_service

test:
	python3 test_api.py --host http://localhost:8000

status:
	docker compose ps

docs:
	@echo "Swagger UI : http://localhost:8000/docs"
	@echo "ReDoc      : http://localhost:8000/redoc"

# ── Docker Hub ─────────────────────────────────────
build:
	docker build -t $(ML_IMAGE):$(TAG)  ./ml_service
	docker build -t $(API_IMAGE):$(TAG) ./api_service
	@echo "✅ Images buildées : $(ML_IMAGE):$(TAG)  $(API_IMAGE):$(TAG)"

push-ml:
	docker push $(ML_IMAGE):$(TAG)

push-api:
	docker push $(API_IMAGE):$(TAG)

push: build push-ml push-api
	@echo "✅ Images pushées sur Docker Hub"
	@echo "   $(ML_IMAGE):$(TAG)"
	@echo "   $(API_IMAGE):$(TAG)"

pull:
	docker pull $(ML_IMAGE):$(TAG)
	docker pull $(API_IMAGE):$(TAG)
	@echo "✅ Images récupérées depuis Docker Hub"
