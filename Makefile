.PHONY: up down logs lint test

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f

lint:
	@echo "Lint targets will be defined as tasks are implemented."

test:
	@echo "Test targets will be defined as tasks are implemented."
