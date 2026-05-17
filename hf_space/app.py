"""Gradio UI wrapping musicgen.generate() for HuggingFace Spaces.

Lets visitors pick seed / genre / backend / output mode, fires one
musicgen.generate() call, and streams the resulting mix.wav back to the
browser. Also surfaces the canonical sample.json so users can inspect the
ground-truth annotations.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

# Repo cloned to /app/musicgen_repo inside the Dockerfile.
REPO_ROOT = "/app/musicgen_repo"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import gradio as gr  # noqa: E402

from config import Config  # noqa: E402
from musicgen import generate, __version__  # noqa: E402

GENRE_CHOICES = ["(any)", "jazz", "hip-hop", "blues", "pop", "electronic", "latin", "reggae", "classical"]
BACKEND_CHOICES = ["markov", "neural"]


def run(seed: int, genre: str, chord_backend: str, melody_backend: str, output_mode: str):
    if seed is None or seed < 0:
        return None, None, "Seed must be a non-negative integer."

    with tempfile.TemporaryDirectory(prefix="musicgen_hf_") as out:
        cli_overrides = {
            "global_seed": int(seed),
            "sample_index": 0,
            "dataset_root": out,
            "output_mode": output_mode,
            "chord_backend": chord_backend,
            "melody_backend": melody_backend,
        }
        if genre and genre != "(any)":
            cli_overrides["genre"] = [genre]

        cfg = Config.load(cli_overrides=cli_overrides)
        result = generate(cfg)

        if result.status != "ok":
            return None, None, f"Generation failed: status={result.status}"

        mix_dst = "/tmp/musicgen_mix.wav"
        if result.mix_path and os.path.exists(result.mix_path):
            with open(result.mix_path, "rb") as src, open(mix_dst, "wb") as dst:
                dst.write(src.read())
        else:
            mix_dst = None

        sj_text = "(no sample.json)"
        if result.sample_json_path and os.path.exists(result.sample_json_path):
            sj_text = pathlib.Path(result.sample_json_path).read_text()

        summary = (
            f"seed={result.seed}  split={result.split}  "
            f"musicality={result.musicality_score:.3f}  "
            f"duration={result.duration_seconds:.2f}s"
        )
        return mix_dst, sj_text, summary


with gr.Blocks(title=f"musicgen {__version__} demo") as demo:
    gr.Markdown(
        f"# musicgen {__version__} — synthetic music dataset generator\n"
        "Pick a seed and (optionally) a genre, click **Generate**, listen to the result, "
        "and inspect the canonical `sample.json` annotation. "
        "Switch the chord/melody backends to `neural` to use the v0.5 LSTM models "
        "(falls back to Markov when models aren't shipped in this Space)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            seed_inp = gr.Number(value=42, precision=0, label="seed")
            genre_inp = gr.Dropdown(GENRE_CHOICES, value="(any)", label="genre")
            chord_inp = gr.Dropdown(BACKEND_CHOICES, value="markov", label="chord_backend")
            mel_inp = gr.Dropdown(BACKEND_CHOICES, value="markov", label="melody_backend")
            mode_inp = gr.Dropdown(
                ["full", "mix-only", "stems-only", "midi-only"],
                value="mix-only",
                label="output_mode",
            )
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            audio_out = gr.Audio(label="mix.wav", type="filepath")
            summary_out = gr.Textbox(label="Result summary", lines=1)
            sj_out = gr.Code(label="sample.json", language="json", lines=20)

    run_btn.click(
        run,
        inputs=[seed_inp, genre_inp, chord_inp, mel_inp, mode_inp],
        outputs=[audio_out, sj_out, summary_out],
    )

    gr.Markdown(
        "**Reproducibility:** Same `seed` + same backends → bit-identical MIDI and "
        "`sample.json` across runs (FluidSynth binary version-locked for WAV identity). "
        "See [the README](https://github.com/dobidu/layered_music_gen) for the full feature set."
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
