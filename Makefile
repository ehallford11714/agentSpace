.PHONY: up down logs lint test db-migrate db-upgrade

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

db-migrate:
	alembic -c backend/alembic.ini revision --autogenerate -m "$(name)"

db-upgrade:
	alembic -c backend/alembic.ini upgrade head
