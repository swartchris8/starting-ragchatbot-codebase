#!/bin/bash
# Run all linting checks

echo "🔍 Running code quality checks..."

echo "Checking code formatting with Black..."
uv run black --check backend/ main.py

echo "Checking import sorting with isort..."
uv run isort --check-only backend/ main.py

echo "Running Flake8 linting..."
uv run flake8 backend/ main.py

echo "Running MyPy type checking..."
uv run mypy backend/ main.py

echo "✅ All quality checks complete!"