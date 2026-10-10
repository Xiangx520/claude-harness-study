# claude-harness-study

this repo for reproducing claude harness

## Local configuration

Copy `.env.example` to `.env`, then fill in the API key, optional API base URL,
and model ID locally. Keep real credentials only in your local `.env` file.
The example contains no credentials, and `.env` is ignored by Git.

Adding `.gitignore` does not remove credentials from earlier commits.
Revoke/rotate any API key that has already been exposed.

## Task management

Run the agent from the project root with `uv run python src/loop_agent.py`.
The main agent manages persistent tasks through six tools:

- `list_tasks()` and `get_task(task_id)` inspect existing records.
- `create_task(subject, description="")` creates a pending task and returns its ID.
- `update_task(task_id, addBlockedBy)` adds dependencies using returned IDs.
- `claim_task(task_id)` starts a pending task once every dependency is completed.
- `complete_task(task_id)` completes a task owned by the main agent and reports
  tasks newly unblocked by its completion.

Create all task nodes before connecting their dependencies. Tasks are saved as
`.tasks/task_XXXXXXXX.json` and remain available across restarts. This directory
is ignored by Git. Dependencies can only be added while a task is pending and
unowned; missing or damaged dependency records keep the task blocked.

The main agent uses owner `agent` and may have multiple tasks in progress.
Subagents retain only the basic shell and file tools; the separate `task(prompt)`
tool still delegates work to a subagent. The former `todo_write` interface has
been removed. `src/code.py` remains a standalone reference example.

Run the regression tests with `uv run python -m unittest discover -s tests -v`.
