# Launcher spec — "a launcher, not a studio"

**One sentence:** a slim Windows GUI that picks a model + port + a few flags, sets the
fork's measured defaults (including the pinning env var), spawns `llama-server`, and opens
the browser at the port. Nothing else.

## Why it exists (the one thing LM Studio doesn't do)

It ships this fork's *measured configs* as one-click presets — the ncmoe knee, ub knee, and
`GGML_CUDA_REGISTER_HOST=1` (a ~2.2× prefill win almost nobody sets by hand). The chat UI is
already served by `llama-server` at the port; we do not rebuild it.

## Non-goals (hard lines — reject scope creep against this list)

- ❌ No chat UI. Open `llama-server`'s own WebUI in the browser.
- ❌ No model downloading / HF browsing / quantization.
- ❌ No multi-model management, model library, or auto-update.
- ❌ No telemetry.
- ❌ No bundling of llama-server binaries in v0 (point at an existing build; bundle later).

If a request would add any of the above, it goes in a "maybe someday" list, not v0.

## Fields (the whole UI)

| field | control | default |
|---|---|---|
| Model (.gguf) | file picker | (empty) |
| Preset | dropdown (see below) | "gpt-oss-120B / bigger-than-VRAM" |
| Port | number | 8080 |
| `-ngl` (GPU layers) | number | 99 |
| `-ncmoe` (CPU-MoE layers) | number | from preset |
| `-ub` (micro-batch) | number | from preset |
| `-fa` (flash-attn) | checkbox | on |
| Host pinning (`GGML_CUDA_REGISTER_HOST=1`) | checkbox | on |
| Threads `-t` | number | (blank = auto) |
| Ctx `-c` | number | 4096 |
| Extra flags | free text passthrough | (empty) |
| **Launch** | button | — |

**Escape hatch:** the "Extra flags" textbox is appended verbatim to the command line, so any
flag we didn't build a field for still works — this is what keeps us off the upstream flag-churn
treadmill.

## Presets (values from BENCHMARKS.md — one source of truth)

- **gpt-oss-120B / bigger-than-VRAM MoE (RTX 5090-class):** `-ngl 99 -ncmoe 22 -fa 1 -b 4096 -ub 2048`, env `GGML_CUDA_REGISTER_HOST=1`.
- **Fits-in-VRAM (dense or small MoE):** `-ngl 99 -fa 1` (no ncmoe), pinning off.
- **Custom:** everything editable, no preset overrides.

Optional helper (v1, not v0): read GPU VRAM (`nvidia-smi`) and hint an ncmoe that sits under the
cliff. Nice-to-have, not required to ship.

## Behavior

1. Build the command line from fields + preset + extra flags.
2. Set env vars (pinning) for the child process only.
3. Spawn `llama-server.exe` (path configured once, remembered).
4. Show live stdout/stderr in a scrollable log pane (so failures are visible).
5. When the server reports listening, open `http://localhost:<port>` in the default browser.
6. Stop button kills the child process cleanly.

That's the entire app. Target ~a few hundred lines.

## Stack + packaging (chosen for the AV story)

- **C#/WPF, framework-dependent** (managed, unpacked — lowest Defender false-positive risk).
- **Ship as a portable zip** on GitHub Releases with a SHA256. No installer, no UPX, no
  single-file pack (those are what trip AV).
- **Code-sign the exe** before the public "post together" launch (standard Authenticode cert
  removes most SmartScreen friction; EV cert = zero friction if budget allows).
- Requires the .NET Desktop **runtime** on the user's machine (present on most Win11); note it in
  the README, or bundle the runtime later if it's friction.

## Prerequisites on the dev box (as of writing)

- Install the .NET **SDK** (only the runtime is present): `winget install Microsoft.DotNet.SDK.9`.
- Build `llama-server` (not currently built): add it to the CMake target build. It's the launch
  target.

## Milestones

- **v0:** the form above, launches gpt-oss-120B with the preset, opens the browser, log pane,
  stop button. Ships as unsigned portable zip for internal testing.
- **v0.5:** code-signed, SHA256, GitHub Release; README section.
- **v1:** VRAM-aware ncmoe hint; remember last-used config; optional bundled server binary.
