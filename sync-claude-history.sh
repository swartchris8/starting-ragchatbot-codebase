if [ -z "${CLAUDE_PROJECT_DIR}" ]
then
  CLAUDE_PROJECT_DIR=$PWD
  echo "CLAUDE_PROJECT_DIR not set, so setting to pwd: $CLAUDE_PROJECT_DIR"
fi
DOT_CLAUDE_PROJ_DIR="-$(echo "$CLAUDE_PROJECT_DIR" | sed 's|^/||' | sed 's|/$||' | sed 's|/|-|g')"
rsync -rvh --delete --progress --stats $HOME/.claude/projects/$DOT_CLAUDE_PROJ_DIR claude-history