# Manifold Switchboard — One-page Explanation

## What this gives you
- Toggle entire environments on/off without leaving the app.
- Isolation + quotas (GPU/CPU/Mem), session persistence, presets.
- Single HUD & hotkeys; layers attach under namespaced mounts.

## Where the meat lives
- `crates/manifold` — registry, FSM lifecycle, budgets, router, trait.
- `crates/plugins/*` — one crate per environment implementing `EnvPlugin`.
- `crates/scene` — layer manager and placeholders for visual layers.
- `apps/inspector_native` — lab UI stubs (add your GPUI/egui code).
- `schemas` — canonical JSON schemas for manifold & sessions.
- `policies` — example policy files for topology/collapse/primes.

## How to proceed
- Replace the `todo!()`/placeholders with your engines and WGSL pipelines.
- Register real plugins in the inspector main; apply budgets and presets.
- Keep the Manifold API surface stable — treat it like a plugin OS.
