# SPDX-License-Identifier: AGPL-3.0-only
import logging
from threading import Thread

log = logging.getLogger(__name__)


def worker(name, target, error):
    def run():
        try:
            target()
        except Exception as exc:
            log.exception("worker_failed", extra={"worker": name})
            error(name, f"{type(exc).__name__}: {exc}")

    thread = Thread(name=name, target=run, daemon=True)
    thread.start()
    return thread
