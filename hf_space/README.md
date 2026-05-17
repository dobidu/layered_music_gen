---
title: musicgen demo
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# musicgen — HuggingFace Space

Gradio UI wrapping [`musicgen`](https://github.com/dobidu/layered_music_gen) — a deterministic synthetic-music dataset generator. Pick a seed + genre + backend, click Generate, listen to the result, inspect the canonical `sample.json`.

## Deploy

Push the contents of this directory to a HuggingFace Space configured for the **Docker** SDK:

```bash
# One-time
huggingface-cli login
huggingface-cli repo create musicgen-demo --type=space --space_sdk=docker

# Push
git clone https://huggingface.co/spaces/<user>/musicgen-demo
cp Dockerfile app.py README.md musicgen-demo/
cd musicgen-demo && git add . && git commit -m "init" && git push
```

The Space builds the Docker image (≈ 3–5 min cold build), then serves the Gradio app on port 7860.

## Notes

- The image symlinks `FluidR3_GM.sf2` (from the `fluid-soundfont-gm` apt package) into all four `sf/<layer>/` directories. Visitors hear a single GM timbre per layer; for variety, fork this Space and add curated `.sf2` files into the image.
- Neural chord/melody backends in this Space fall back to Markov because trained `.pt` checkpoints aren't shipped — train your own following [`docs/neural-generators.md`](https://github.com/dobidu/layered_music_gen/blob/main/docs/neural-generators.md) and `COPY` them into `/app/musicgen_repo/models/` in the Dockerfile.
- Free CPU Spaces have ~16 GB RAM and 2 vCPUs — one sample takes 5–15 s end-to-end.
