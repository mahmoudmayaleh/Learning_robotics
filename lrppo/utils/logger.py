"""Logging helpers.

Includes simple console logger setup and a CSV-based metrics logger
for recording per-episode training metrics.
"""

import logging
import csv
import os
from typing import List, Dict, Optional


def setup_logger(name='lrppo', level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


class CSVLogger:
    """Minimal CSV logger for episode metrics.

    Writes episodes as rows to a CSV file and optionally prints running
    averages every `print_interval` episodes.
    """

    def __init__(self, csv_path: str, fieldnames: List[str], print_interval: int = 0) -> None:
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        self.print_interval = int(print_interval)
        self.count = 0
        self.history: List[Dict[str, float]] = []

        # ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)

        # create file and write header if not existing
        write_header = not os.path.exists(self.csv_path)
        self._file = open(self.csv_path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log_episode(self, metrics: Dict[str, float]) -> None:
        """Append a metrics dict as a row. Missing keys will be filled with empty strings."""
        row = {k: metrics.get(k, '') for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

        # keep history for running averages
        self.count += 1
        self.history.append(metrics)

        if self.print_interval > 0 and (self.count % self.print_interval == 0):
            self._print_running_averages()

    def _print_running_averages(self) -> None:
        # compute averages for numeric fields
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for row in self.history[-self.print_interval :]:
            for k, v in row.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                sums[k] = sums.get(k, 0.0) + fv
                counts[k] = counts.get(k, 0) + 1
        avgs = {k: sums[k] / counts[k] for k in sums if counts.get(k, 0) > 0}
        print(f"[CSVLogger] last {self.print_interval} episodes averages: {avgs}")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


__all__ = ["setup_logger", "CSVLogger"]
