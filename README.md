# Jev Claude Gateway

A local gateway that sits between Claude Code and Anthropic. For each new task it asks
TypeSafe's Jev (through OpenRouter) how demanding the task is, then a local policy picks
Haiku, Sonnet or Opus. Jev only classifies; it never answers your request.

```
Claude Code (terminal) -> gateway :8787 -> Jev classifies -> policy -> Haiku / Sonnet / Opus -> Anthropic
```

## Day to day

| Command | What it does |
|---|---|
| `/jev on` (inside Claude Code) | Jev picks the model for each task |
| `/jev off` | Pass-through: requests go to the model Claude Code chose |
| `/jev shadow` | Ask Jev and log its choice, but change nothing |
| `/jev status` | Show the current mode |
| `./jevstats [hours]` | Summary of the decision log (counts per tier, fallbacks, Jev cost) |
| `./jev-models-check` | Report new Claude models on Anthropic's public page (changes nothing) |

The decision log is `~/.jev-gateway/decisions.jsonl` (no keys, no message text).

## Where it works
- Claude Code in the **terminal**: yes.
- Claude Code in the **desktop app's Code tab**: no. The app sets its own `ANTHROPIC_BASE_URL`, which overrides the setting.
- Normal chat and Cowork: no, they never call a local gateway.

## If Claude Code can't connect
1. `./jev-restart` restarts the background gateway.
2. If that fails, `./jev-bypass` removes the gateway setting so Claude Code talks straight to Anthropic (start a new session). `./jev-bypass connect` puts it back. A backup of your settings is saved as `~/.claude/settings.json.jev-backup`.
Run these in a plain terminal, not from a Claude chat.
Note: `/jev off` does not help if the gateway is down; it talks to the gateway itself.

## How it is wired
- `gateway.py`: forwards everything to Anthropic; inspects only `POST /v1/messages`; control endpoints (`/jev/status`, `/jev/mode`) reject web pages; `/health`.
- `routing.py`: finds the human's latest message (ignores tool results and injected reminders), asks Jev (3s timeout), applies the policy, adapts the request for the chosen model, bumps up a tier when the conversation is near Haiku's input limit. Any Jev failure or low confidence falls back to Sonnet.
- `models/registry.json`: the models routing may use, with their limits and unsupported features. New models are added here by hand, never automatically.
- Safety net: if a switched request is rejected (400/404/422) the gateway retries once on the model Claude Code asked for.
- `launchd/com.yuvathefirst.jev-gateway.plist`: runs the gateway at login and restarts it if it crashes (installed to `~/Library/LaunchAgents/`).
- The mode (`off`/`shadow`/`on`) is saved in `~/.jev-gateway/mode`. It starts as `off` if never set.

## The key
The Jev (OpenRouter) key is read from `OPENROUTER_API_KEY`, else from the Keychain item `jev-openrouter`
(a background job can't see shell variables). Store it with:
`security add-generic-password -U -a "$USER" -s jev-openrouter -w "$OPENROUTER_API_KEY"`
Never put the key in a file in this folder.

## Adding a new model
1. `./jev-models-check` shows what's new. 2. Edit `models/registry.json`: model id, `max_output_tokens`,
`max_input_tokens`, `drop_betas`, `supports_*` flags (find them from the API's error messages).
3. Test with `JEV_FORCE_TIER=<tier>` on a spare port. 4. `./jev-models-check --mark-seen`.

## Settings (environment variables)
`JEV_ROUTE_MODE`, `JEV_MIN_CONFIDENCE` (0.6), `JEV_TIMEOUT_S` (3), `JEV_SECURITY_THRESHOLD` (0.7),
`JEV_MODEL`, `JEV_URL`, `JEV_STATE_DIR`, `JEV_FORCE_TIER` (testing only).
