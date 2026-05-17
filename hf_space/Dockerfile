# HuggingFace Space — Dockerfile-backed.
# Builds a Gradio UI around musicgen.generate() so anyone with the Space URL
# can pick a seed/genre/backend and get back an audio sample.
#
# Reference: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

# System deps — FluidSynth binary + GM soundfont + ffmpeg for audio post-processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fluidsynth \
        fluid-soundfont-gm \
        ffmpeg \
        libsndfile1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires the container to write to /tmp (or a mounted dir).
WORKDIR /app

# Install musicgen (with neural extra) from the public repo.
RUN pip install --no-cache-dir "musicgen[neural] @ git+https://github.com/dobidu/layered_music_gen.git" \
    gradio>=4.0

# Clone the repo so genres/, chord_patterns.txt, etc. are available next to app.py.
RUN git clone https://github.com/dobidu/layered_music_gen.git /app/musicgen_repo

# Wire the apt-installed FluidR3_GM into the per-layer sf/ dirs.
RUN for layer in beat melody harmony bassline; do \
        mkdir -p /app/musicgen_repo/sf/$layer && \
        ln -sf /usr/share/sounds/sf2/FluidR3_GM.sf2 /app/musicgen_repo/sf/$layer/FluidR3_GM.sf2 ; \
    done

COPY app.py /app/app.py

# HF Spaces default port is 7860.
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "/app/app.py"]
