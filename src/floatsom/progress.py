"""Show the floatsom progress logs during a call with verbose=True."""

import logging
from contextlib import contextmanager


@contextmanager
def progress_logging(enabled: bool):
    """Let the INFO logs of floatsom through during the block, if ``enabled``.

    Add a handler that prints to stderr only when the program has no log handler,
    so a program that configures logging keeps its own format. At the end of the
    block, restore the logger level and remove the handler.
    """
    if not enabled:
        yield
        return
    logger = logging.getLogger("floatsom")
    old_level = logger.level
    handler = None
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s floatsom: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    try:
        yield
    finally:
        logger.setLevel(old_level)
        if handler is not None:
            logger.removeHandler(handler)
