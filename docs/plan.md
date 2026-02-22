  Mode Detection (benchmark/core/config.py)
  ┌───────────────────────────────┬────────────────┐
  │         Config Field          │      Mode      │
  ├───────────────────────────────┼────────────────┤
  │ eval_results set              │ SKILLFLOW_EVAL │
  ├───────────────────────────────┼────────────────┤
  │ mcp_url set                   │ MCP            │
  ├───────────────────────────────┼────────────────┤
  │ skills.skillflow_peer_url set │ SKILLFLOW      │
  ├───────────────────────────────┼────────────────┤
  │ skills.skills_dir set         │ SKILLS         │
  ├───────────────────────────────┼────────────────┤
  │ nothing                       │ BASELINE       │
  └───────────────────────────────┴────────────────┘

  ---
  1. SKILLS — Static tar.gz injection

  Pre-selected skill folders are archived and uploaded into the container:

  1. SkillManager finds all SKILL.md folders, filters by curated list or task name
  2. TarGzSkillInjector stages them → skills.tar.gz → uploads to container → extracts at /logs/agent/skills/
  3. Agent gets a Jinja2 instruction template telling it to cat $CODEX_HOME/skills/<name>/SKILL.md

  2. SKILLFLOW — Dynamic HTTP peer

  No pre-injection. A skillflow-client script is uploaded to the container:

  - skillflow-client search "query" → hits HTTP peer /skills?query=...
  - skillflow-client get <id> / save <id> → downloads SKILL.md on demand

  The agent must proactively search before coding.

  3. MCP — Native tool via MCP server

  Registers an MCP server with Codex (codex mcp add skillflow --url {mcp_url}), which exposes a retrieve_skill() tool. Codex discovers it automatically from the tool
  description.

  4. SKILLFLOW_EVAL — Your current config

  This is what skillflow-eval.json uses. It combines injection + MCP:

  1. Reads eval_results JSON → resolves skill keys like skillsbench/{task_id}/{skill_name} to actual folders under tasks_dir_for_skills
  2. Injects those skills via tar.gz into /logs/agent/skills/
  3. Registers the MCP server at mcp_url — the MCP retrieve_skill() tool returns the injected skill paths so the agent knows what to read

  So for your config: the eval-selected skills get physically injected into the container, and the MCP server tells the agent where to find them.
