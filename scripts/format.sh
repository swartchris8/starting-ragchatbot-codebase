#!/bin/bash
# Format all Python code with black and isort

echo "🎨 Formatting Python code..."

echo "Running Black formatter..."
uv run black backend/ main.py

echo "Running isort for import sorting..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"