#!/usr/bin/env bash
set -euo pipefail

# Настрой под себя при необходимости:
REMOTE_URL=${REMOTE_URL:-git@github-2probe:ilyamus74-ctrl/predict.git}
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

# 1) Инициализация репозитория (если .git нет)
if [ ! -d ".git" ]; then
  git init
fi

# 2) Локальная идентичность (только для этого репо)
if ! git config user.email >/dev/null; then
  git config user.email "ilya@example.local"
fi
if ! git config user.name >/dev/null; then
  git config user.name "Illia"
fi

# 3) Убедимся, что текущая ветка = main (создадим, если «unborn»)
current_branch=$(git symbolic-ref --short -q HEAD || true)
if [ -z "${current_branch}" ]; then
  # репо без коммитов
  git checkout -B "$DEFAULT_BRANCH"
else
  # переименуем текущую в main (если нужна консолидация)
  git branch -M "$DEFAULT_BRANCH"
fi

# 4) origin → на нужный URL (через алиас github-2probe, чтобы взялся правильный ключ)
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

# 5) Индексация и коммит
git add -A
if ! git diff --cached --quiet; then
  git commit -m "sync local -> remote"
else
  # Если это самый первый пуш и коммитов нет — сделаем пустой initial
  if ! git rev-parse HEAD >/dev/null 2>&1; then
    git commit --allow-empty -m "Initial commit"
  fi
fi

# 6) Аккуратный пуш: с lease только если удалённая main существует
set +e
REMOTE_MAIN_HASH=$(git ls-remote --heads origin "$DEFAULT_BRANCH" | awk '{print $1}')
set -e

if [ -n "$REMOTE_MAIN_HASH" ]; then
  git push -u origin HEAD:"$DEFAULT_BRANCH" --force-with-lease="$DEFAULT_BRANCH:$REMOTE_MAIN_HASH"
else
  git push -u origin HEAD:"$DEFAULT_BRANCH"
fi

echo "✅ Done: pushed to $REMOTE_URL ($DEFAULT_BRANCH)"