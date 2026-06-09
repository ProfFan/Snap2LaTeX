# Snap2LaTeX (standalone macOS app)

A macOS menu-bar app that turns a screenshot of a math equation into LaTeX. It
uses [nougat-latex-ocr](https://github.com/NormXU/nougat-latex-ocr) for
recognition and runs on Apple Silicon via MPS by default.

> End users: just grab a prebuilt `.app` from
> [releases](https://github.com/ProfFan/Snap2LaTeX/releases) — no Python needed.
> Everything below is for **building from source**.

## Prerequisites

- macOS on Apple Silicon (MPS) — Intel falls back to CPU.
- [uv](https://docs.astral.sh/uv/) for Python + dependency management:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

That's it. uv installs the right Python (pinned in `.python-version`) and all
dependencies itself — no manual `pip`, no global Python setup.

## Dependencies

Dependencies are declared in `pyproject.toml` and pinned to exact versions in
`uv.lock` (committed). The lockfile guarantees everyone builds against the same
versions — no more drift from an unpinned `requirements.txt`.

- Runtime deps live under `[project.dependencies]`.
- The build tool (`pyinstaller`) lives in the `dev` dependency group, which
  `uv sync` installs by default.

## Build

```bash
cd standalone_app

# 1. Create the venv and install everything from the lockfile.
uv sync

# 2. Build the .app bundle.
uv run pyinstaller Snap2LaTeX.spec --noconfirm
```

The bundle is written to `dist/Snap2LaTeX.app`. Drag it to `/Applications` to
install. (`dist/Snap2LaTeX/` is the same program as a plain folder, handy for
debugging.)

## Run from source (without building)

```bash
uv run python Snap2LaTeX.py
```

On first launch the app downloads the ~1.4 GB `Norm/nougat-latex-base` model
from Hugging Face into `~/.cache/huggingface`. This takes a few minutes and can
look like a hang — it isn't. Subsequent launches are fast.

## Usage

A `∑`-style icon appears in the menu bar. Click it → **Capture**, then select
the screen region containing the equation. The recognized LaTeX is shown and
**automatically copied to the clipboard**. The icon turns yellow while
processing.

## Managing dependencies

```bash
uv add <pkg>            # add a runtime dependency (updates pyproject + uv.lock)
uv add --dev <pkg>      # add a build/dev-only dependency
uv lock --upgrade       # re-resolve to newest allowed versions
uv sync                 # apply the lockfile to the venv
```

Always commit `pyproject.toml` **and** `uv.lock` together.

## Troubleshooting

- **`'MBartConfig' object has no attribute 'max_length'`** — transformers 5.x
  moved `max_length` out of the model config. Already handled in code via a
  `getattr(..., 800)` fallback; make sure you're building current source.
- **Missing `transformers` submodules / `NougatConfig` not found in the bundled
  app** — PyInstaller can't follow transformers' lazy (`_LazyModule`) imports.
  `Snap2LaTeX.spec` works around this with `collect_submodules('transformers')`
  and `collect_data_files('transformers')`; don't remove those.
- **App launches then immediately quits** — usually the model download failed or
  was interrupted. Run from source (`uv run python Snap2LaTeX.py`) to see logs.
