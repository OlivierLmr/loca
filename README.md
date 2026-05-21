# loca

Wake-on-request local LLM proxy with a live TUI. Run multiple MLX models
simultaneously on a single OpenAI-compatible endpoint, then point clients
like [smelt](https://github.com/leonardcser/agent) or
[opencode](https://opencode.ai) at it. loca handles process lifecycle,
idle eviction, and routing.

```
┌─ Proxy  http://127.0.0.1:11431/v1   uptime 12m 3s   34 req   0 err ─┐
│  RAM    15 / 24 GB   2 models loaded                                │
├─ Models ────────────────────────────────────────────────────────────┤
│  qwen3.5-9b       ~6 GB   READY    23s / 10m   yes   Qwen 3.5 9B …  │
│  qwen3.5-27b      ~16 GB  —        —           no    Qwen 3.5 27B…  │
│  phi4-14b         ~9 GB   READY    1m / 10m    yes   Phi-4 14B …    │
├─ Log: qwen3.5-9b ───────────────────────────────────────────────────┤
│  12:01:51 ready in 5.7s                                              │
│  12:01:52 POST /v1/chat/completions stream=true                      │
│  …                                                                    │
├─ Log: loca ─────────────────────────────────────────────────────────┤
│  12:00:00 listening on 127.0.0.1:11431                              │
│  12:01:45 loading qwen3.5-9b                                         │
├─────────────────────────────────────────────────────────────────────┤
│ q Quit  Tab Cycle pane  ↑↓ Nav  ⏎ Load/Stop  s Stop  t +10m  …      │
└─────────────────────────────────────────────────────────────────────┘
```

## What it gives you

- **Multi-model.** Loads multiple `mlx_lm.server` backends, each on its own
  port. The proxy reads each request's `model` field and routes to the right
  process — so different clients (or different sessions) can use different
  models concurrently.
- **Idle eviction per model.** Configurable per-instance from the TUI
  (`t` +10m, `T` +1h, `r` reset 10m).
- **No daemon.** loca runs in the foreground. When you `q` or Ctrl-C, all
  spawned backends are killed and RAM is freed. No launchd, no systemd, no
  surprises after reboot.
- **OpenAI-compatible API.** Drop-in for anything that speaks the OpenAI
  Chat Completions / Completions API.
- **Single-instance.** Refuses to start if another loca is already bound to
  the listen port.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4). MLX is Apple-only.
- **Rust toolchain** for `cargo install`. The simplest path on macOS:

  ```sh
  brew install rust
  ```

  If you want multi-version management or nightly support later, install
  `rustup` instead: `brew install rustup-init && rustup-init -y`. The
  official upstream installer at https://rustup.rs/ uses a curl-piped shell
  script — it's what the Rust team publishes, but `brew` is more conventional
  on macOS.

- **mlx-lm** (Python) for the actual model serving. Install with `uv`:

  ```sh
  brew install uv      # if you don't have it
  uv tool install mlx-lm
  ```

  This puts `mlx_lm.server` on your PATH.

- 16 GB RAM minimum; 32+ GB recommended for the larger models in the default
  alias list.

## Install

```sh
cargo install --git https://github.com/OlivierLmr/loca
```

This drops the `loca` binary into `~/.cargo/bin`. Make sure that's on your
PATH (it usually is if you used rustup).

To upgrade later:

```sh
cargo install --git https://github.com/OlivierLmr/loca --force
```

## First run

```sh
loca
```

On first launch, loca creates everything it needs:

| Path | Purpose |
|------|---------|
| `~/.config/loca/models.toml` | Alias map. Edit to add/remove models. |
| `~/.config/loca/proxy.toml`  | Optional overrides (commented template). |
| `~/.local/state/loca/`       | Runtime state directory. |
| `~/.cache/huggingface/hub/`  | Created if missing (mlx_lm scans it). |

It then starts the foreground TUI. The proxy is now live on
`http://127.0.0.1:11431/v1`.

## TUI key bindings

**Global** (any pane):

| Key | Action |
|-----|--------|
| `q` or `Ctrl-c` | Quit (kills all backends) |
| `Tab` | Cycle focus through Models → Model log → loca log |

**Models pane:**

| Key | Action |
|-----|--------|
| `↑` / `↓` | Move row cursor |
| `Enter` | Load the selected model (or stop it if already loaded) |
| `s` | Stop the selected model |
| `t` | +10 min to its idle timeout |
| `T` | +1 hour to its idle timeout |
| `r` | Reset its idle timeout to 10 min |

**Log panes** (Model log or loca log):

| Key | Action |
|-----|--------|
| `c` | Clear that pane's history (widget + log bus) |

When focus is on the models pane, the footer shows the model-pane shortcuts.
When focus is on a log pane, it shows just `c Clear log`. This is intentional —
each pane only handles the keys that make sense for it.

## Configuration

### `~/.config/loca/models.toml`

Each entry maps an alias (what your clients send as the `model` field) to an
MLX-format model on Hugging Face. Quote aliases containing dots, since TOML
treats unquoted dots as table nesting.

```toml
[models."qwen3.5-9b"]
hf = "mlx-community/Qwen3.5-9B-MLX-4bit"   # required: HF repo id
ram_gb = 6                                  # informational, used by RAM meter
description = "Qwen 3.5 9B — general, fast" # shown in the TUI table
```

Add or remove entries freely. Changes are picked up on the next `loca` start.

### `~/.config/loca/proxy.toml` (optional)

All keys are optional. The file is written as a commented template on first
run so you can see the defaults:

```toml
[proxy]
# listen_host = "127.0.0.1"
# listen_port = 11431
# backend_host = "127.0.0.1"
# backend_port_start = 11432   # first port handed to a backend
# backend_port_count = 16      # how many ports to reserve
# idle_timeout_sec = 600       # default per-instance idle timeout (10 min)
# spawn_timeout_sec = 240      # max seconds to wait for a backend to come up
# max_tokens = 16384           # ceiling for `mlx_lm.server --max-tokens`
# ram_budget_gb = 24           # advisory; the TUI shows red over the budget
```

### Environment variable

- `LOCA_KEY` — Some clients require an API key value to populate the
  Authorization header. mlx_lm.server doesn't authenticate, so set this to
  any non-empty string:

  ```sh
  # in ~/.zshrc
  export LOCA_KEY=loca-dummy
  ```

## Wiring up clients

### smelt

[smelt](https://github.com/leonardcser/agent) is a Rust coding-agent TUI.
Add a provider block to `~/.config/smelt/init.lua`:

```lua
smelt.provider.register("Local", {
  type = "openai-compatible",
  api_base = "http://127.0.0.1:11431/v1",
  api_key_env = "LOCA_KEY",
  models = {
    { name = "qwen3.5-9b"        }, -- ~6 GB
    { name = "qwen3.5-27b"       }, -- ~16 GB
    { name = "qwen3.5-35b-a3b"   }, -- ~20 GB (MoE)
    { name = "phi4-14b"          }, -- ~9 GB
    { name = "gpt-oss-20b"       }, -- ~12 GB
    { name = "mistral-small-24b" }, -- ~14 GB
  },
})
```

Then start a session with:

```sh
smelt -m qwen3.5-9b
```

### opencode

[opencode](https://opencode.ai) is an agentic coding terminal. Add to
`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local (loca)",
      "options": {
        "baseURL": "http://127.0.0.1:11431/v1",
        "apiKey": "loca-dummy"
      },
      "models": {
        "qwen3.5-9b":        { "name": "Qwen 3.5 9B — ~6 GB — general, fast" },
        "qwen3.5-27b":       { "name": "Qwen 3.5 27B — ~16 GB — general/reasoning" },
        "qwen3.5-35b-a3b":   { "name": "Qwen 3.5 35B-A3B (MoE) — ~20 GB" },
        "phi4-14b":          { "name": "Phi-4 14B — ~9 GB — reasoning" },
        "gpt-oss-20b":       { "name": "gpt-oss 20B — ~12 GB" },
        "mistral-small-24b": { "name": "Mistral Small 24B — ~14 GB" }
      }
    }
  }
}
```

Pick "Local (loca) / qwen3.5-9b" in opencode's model menu, or use:

```sh
opencode --model local/qwen3.5-9b
```

### Any OpenAI-compatible client

Point it at `http://127.0.0.1:11431/v1` with any non-empty API key, and use
one of the aliases from your `models.toml` as the model name.

## HTTP API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET    | `/v1/models` | — | List of aliases with `loaded: bool` per entry |
| POST   | `/v1/chat/completions` | OpenAI request body | Streams (SSE) or JSON, routed by `model` |
| POST   | `/v1/completions` | OpenAI request body | Same |
| GET    | `/admin/status` | — | JSON: loaded instances, RAM use, budget |
| POST   | `/admin/stop` | optional `{model: alias}` (or `?model=alias`) | Stops one or all |
| POST   | `/admin/preload` | `{model: alias}` (or `?model=alias`) | Warms a model without sending a real request |
| GET    | `/health` | — | `{ok: true}` |

The proxy strips and rewrites the `model` field of forwarded requests so the
backend sees the HF id it was spawned with.

## CLI

| Command | Description |
|---------|-------------|
| `loca`            | Start the TUI + proxy (default) |
| `loca ls`         | List aliases from models.toml (works offline) |
| `loca status`     | One-shot JSON status (requires loca running) |
| `loca preload <alias>` | Ask the running loca to warm a model |
| `loca init`       | Force-create config dirs + defaults |
| `loca help`       | Help text |

## Architecture

- **BackendInstance** owns one `mlx_lm.server` subprocess. Its stdout/stderr
  are streamed into a per-alias log bus.
- **BackendManager** holds a `HashMap<alias, BackendInstance>` and a port pool.
  An idle watcher task evicts instances that exceed their per-instance idle
  timeout.
- **Proxy** is an `axum` router. Forwarded streaming responses are passed
  through with `reqwest::Response::bytes_stream()` → `axum::body::Body::from_stream(_)`,
  so there's no buffering of SSE chunks.
- **TUI** is `ratatui` driven from a `tokio::select!` loop over crossterm
  events, log broadcasts, and a 1-second tick.

Source layout:

```
src/
├── main.rs       # clap dispatch
├── paths.rs      # known filesystem locations
├── config.rs     # serde models for the TOML files
├── init.rs       # first-run setup
├── log_bus.rs    # broadcast + per-source history
├── backend.rs    # BackendInstance
├── manager.rs    # BackendManager
├── proxy.rs      # axum routes
└── tui.rs        # TUI state machine + draw
```

## Troubleshooting

**`port 11431 is already in use`** — another loca is already running (or
some other process grabbed the port). Find it with `lsof -nP -i :11431` and
either kill it or change `listen_port` in `~/.config/loca/proxy.toml`.

**Backend keeps stopping mid-response** — the request was capped by
`max_tokens`. Raise it in `~/.config/loca/proxy.toml` and restart loca.

**Long traceback ending in `BrokenPipeError`** in the model log — your client
disconnected mid-stream and `mlx_lm.server` doesn't handle the broken socket
gracefully. Cosmetic, can be ignored.

**Gated HuggingFace models** (Llama, Gemma) — run `huggingface-cli login`
once. The default alias list ships open models only, so this only matters
if you add a gated model yourself.

**First request to a model hangs** — it's downloading multi-GB weights from
HF. Watch progress in the model log pane; it's a one-time cost per model.

## License

MIT. See [LICENSE](LICENSE).
