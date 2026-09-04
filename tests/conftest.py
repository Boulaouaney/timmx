import os

# Wide help output so option names are not truncated in CLI help tests on narrow runners.
# Typer reads this at import time, so it must be set before any test module imports typer.
os.environ.setdefault("TERMINAL_WIDTH", "200")
