"""Neural backends for chord and melody generation (v0.5).

All torch usage is guarded behind ``HAS_TORCH``. When torch is absent the
neural backend falls back to Markov silently.
"""
try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
