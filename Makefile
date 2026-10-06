dev-backend:
	cd backend && uv run uvicorn agentforge.api:app --reload

dev-frontend:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest -q

up:
	docker compose up --build

down:
	docker compose down

.PHONY: dev-backend dev-frontend test up down
