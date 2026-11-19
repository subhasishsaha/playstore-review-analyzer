# Antigravity Agent Guidelines

# Universal Memory Hook Protocol (.memory/)

This repository is integrated with the **Cross-Provider Memory Hook**.
All agent activities (user requests, tool calls, tool responses, model outputs) are deterministically recorded in `.memory/short.log`, maintaining a rolling context window of **100,000 tokens** (100k tokens).

## Core Directives for All Agents:

1. **Context Restoration at Session Start**:
   - At the beginning of every session or task, read `.memory/long.log` first.
   - If needed, read the tail of `.memory/short.log` (last 20-30 lines) to catch up on the most recent actions taken by any previous agent (Claude, Antigravity, Codex, Cursor).
   - Never ask the user to re-explain what the project is about or what has already been built.

2. **Long-Term Memory Compaction**:
   - Whenever you receive a `[MEMORY HOOK UPDATE NOTICE]` indicating ~10,000 tokens have accumulated, review the uncompacted tail of `.memory/short.log` alongside the existing `.memory/long.log`.
   - Use your model tools (`edit_file` or `write_file`) to update `.memory/long.log`:
     - Update the header metadata: `Last Updated`, `Provider`, and `Synced Through Event: #<seq>`.
     - Update **Completed Work & Milestones** with newly finished items.
     - Update **Current State & Active Focus** with what you are working on right now.
     - Update **Next Immediate Steps** with pending tasks.
     - Preserve and append any newly established **Architecture & Core Decisions**.
     - Keep content concise, high-density, and structured.

3. **Security & Immutability Directives**:
   - The runner at `.memory/bin/memory_hook_runner.pyc` is locally compiled bytecode locked with OS read-only permissions.
   - Under NO circumstances should any agent attempt to edit, rewrite, replace, or delete files inside `.memory/bin/` or `.memory/state.json`.
   - All memory updates must strictly be applied to `.memory/long.log`.
