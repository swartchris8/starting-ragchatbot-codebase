#!/usr/bin/env python3
"""Development script for running code quality checks."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} passed")
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Run code quality checks."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--fix", action="store_true", help="Fix formatting issues automatically"
    )
    parser.add_argument(
        "--check",
        choices=["black", "isort", "flake8", "mypy", "all"],
        default="all",
        help="Run specific check",
    )
    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"Running quality checks from: {project_root}")

    success = True
    
    if args.check in ("black", "all"):
        if args.fix:
            success &= run_command(
                ["uv", "run", "black", "backend/", "main.py"], "Black formatting"
            )
        else:
            success &= run_command(
                ["uv", "run", "black", "--check", "backend/", "main.py"],
                "Black format check",
            )

    if args.check in ("isort", "all"):
        if args.fix:
            success &= run_command(
                ["uv", "run", "isort", "backend/", "main.py"], "Import sorting"
            )
        else:
            success &= run_command(
                ["uv", "run", "isort", "--check-only", "backend/", "main.py"],
                "Import sort check",
            )

    if args.check in ("flake8", "all"):
        success &= run_command(
            ["uv", "run", "flake8", "backend/", "main.py"], "Flake8 linting"
        )

    if args.check in ("mypy", "all"):
        success &= run_command(
            ["uv", "run", "mypy", "backend/", "main.py"], "MyPy type checking"
        )

    if success:
        print("\n🎉 All quality checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some quality checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()