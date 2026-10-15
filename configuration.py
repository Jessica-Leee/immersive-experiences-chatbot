"""Resolve configuration without requiring a Streamlit secrets file."""

import os


def resolve_api_key(name, secrets=None):
    if secrets is not None:
        try:
            value = secrets.get(name)
        except FileNotFoundError:
            # Streamlit 1.31 raises this when no secrets.toml is configured.
            pass
        else:
            if value:
                return value
    return os.getenv(name)
