"""Show the floatsom progress logs when a user sets verbose=True."""

import logging


def show_progress() -> None:
    """Let the INFO logs of floatsom through.

    Add a handler that prints to stderr only when the program has no log handler,
    so a program that configures logging keeps its own format.
    """
    logger = logging.getLogger("floatsom")
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s floatsom: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
