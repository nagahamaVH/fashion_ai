import sys
from io import StringIO
import re
import numpy as np


def format_coco_summarize(text):
    text = re.sub(r"(.*?)\n", "", text, count=1)  # clean first line (header)
    metrics = {}
    while text != "":
        found = re.search(r" (.*?) = (-*\d.\d{3})\n", text)
        metrics[found.group(1)] = np.float32(found.group(2))
        text = re.sub(r"(.*?)\n", "", text, count=1)  # update text

    return metrics


def get_metrics(coco_evaluator):
    stdout_backup = sys.stdout
    sys.stdout = string_buffer = StringIO()
    coco_evaluator.summarize()
    sys.stdout = stdout_backup  # restore old sys.stdout
    raw_metrics = string_buffer.getvalue()
    metrics = format_coco_summarize(raw_metrics)

    return metrics
