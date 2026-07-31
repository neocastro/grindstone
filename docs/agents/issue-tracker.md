# Issue Tracker

Issues live in **GitHub Issues** on this repo (`neocastro/grindstone`).

- CLI: `gh issue create`, `gh issue list`, `gh issue view`
- Work units are the RAG-1..RAG-8 tickets (migrated from the tlarc tracker —
  see grindstone#1 for the parent PRD)
- An issue is AFK-ready when it carries the `ready-for-agent` label and its
  body satisfies the acceptance criteria template
- The weak local model grinds `ready-for-agent` issues; the strong agent
  writes them, reviews diffs, and handles HITL items
- The grinding runner lives in the sibling tlarc repo
  (`scripts/grind-next-issue.sh`), which calls `gs build-prompt` to construct
  the agent prompt safely
