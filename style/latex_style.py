"""LaTeX table utilities: format numbers, bold best values, emit booktabs tables."""
from typing import Sequence, Optional
import math

def fmt(value, decimals=2, percent=False):
    if math.isnan(value): return "--"
    return f"{value*100:.{decimals}f}" if percent else f"{value:.{decimals}f}"

def bold(s): return r"\textbf{" + s + "}"

def best_in_col(values, higher_is_better=True, decimals=2, percent=False):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean: return ["--"] * len(values)
    best_val = max(clean) if higher_is_better else min(clean)
    out = []
    for v in values:
        if v is None or math.isnan(v): out.append("--")
        else:
            s = fmt(v, decimals=decimals, percent=percent)
            out.append(bold(s) if v == best_val else s)
    return out

def make_table(headers, rows, caption="", label="", col_format=None):
    n = len(headers)
    if col_format is None: col_format = "l" + "c" * (n - 1)
    lines = [
        r"\begin{table}[t]", r"  \centering", r"  \small",
        rf"  \begin{{tabular}}{{{col_format}}}",
        r"    \toprule",
        "    " + " & ".join(headers) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if caption: lines.append(rf"  \caption{{{caption}}}")
    if label:   lines.append(rf"  \label{{tab:{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
