#!/usr/bin/env bash
set -euo pipefail

# Deploy script for sciencestack CLI
# Usage: ./deploy.sh [patch|minor|major]

BUMP_TYPE="${1:-patch}"
PYPROJECT="pyproject.toml"
INIT_FILE="src/sciencestack/__init__.py"

# Get current version from pyproject.toml
CURRENT=$(grep '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT"

# Compute next version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
case "$BUMP_TYPE" in
  patch) PATCH=$((PATCH + 1)) ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  *) echo "Usage: ./deploy.sh [patch|minor|major]"; exit 1 ;;
esac
NEXT="$MAJOR.$MINOR.$PATCH"
echo "Bumping to: $NEXT"

# Confirm
read -rp "Proceed? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# Bump version in pyproject.toml
sed -i '' "s/^version = \"$CURRENT\"/version = \"$NEXT\"/" "$PYPROJECT"

# Sync version in __init__.py
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$NEXT\"/" "$INIT_FILE"

echo "Updated $PYPROJECT and $INIT_FILE"

# Clean and build
rm -rf dist/
python -m build
echo "Built dist/"

# Upload to PyPI
python -m twine upload dist/*
echo "Uploaded to PyPI"

# Git commit, tag, push
git add "$PYPROJECT" "$INIT_FILE"
git commit -m "bump version to $NEXT"
git tag "v$NEXT"
git push && git push --tags

echo "Done! Published sciencestack v$NEXT"
echo "https://pypi.org/project/sciencestack/$NEXT/"
