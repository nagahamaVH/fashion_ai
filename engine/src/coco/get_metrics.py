import sys
from io import StringIO
import re


def format_coco_summarize(text):
    metrics = {}
    while text != "":
        metric_found = re.search(r"^IoU metric: (.*?)\n", text)
        if metric_found is not None:
            metric_name = metric_found.group(1)
            text = re.sub(r"(.*?)\n", "", text, count=1)  # clean metric line
            metric_found = None
        found = re.search(r" (.*?) = (-*\d+.\d+)\n", text)
        metrics[metric_name + " - " + found.group(1)] = float(found.group(2))
        text = re.sub(r"(.*?)\n", "", text, count=1)  # update tex

    return metrics


def get_metrics(coco_evaluator):
    stdout_backup = sys.stdout
    sys.stdout = string_buffer = StringIO()
    coco_evaluator.summarize()
    sys.stdout = stdout_backup  # restore old sys.stdout
    raw_metrics = string_buffer.getvalue()
    metrics = format_coco_summarize(raw_metrics)

    return metrics


if __name__ == "__main__":
    import json

    with open("./report.txt", "r") as f:
        text = f.read()
    metrics = format_coco_summarize(text)
    with open("./metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False)
