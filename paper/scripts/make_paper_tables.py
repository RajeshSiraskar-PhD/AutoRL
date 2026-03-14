"""Generate LaTeX-ready tables from the paper CSV artefacts.

Scope: Only reads files under the paper/ folder.

Outputs:
- tables/ieee_unseen_summary.tex
- tables/ieee_best_attention_per_algo.tex
- tables/hypothesis_tests_summary.tex

Run:
  python3 scripts/make_paper_tables.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
IEEE_EVALS = PAPER_DIR / "figures and reports" / "IEEE_Results" / "_Evals_Results_IEEE.csv"
IEEE_HYP = PAPER_DIR / "figures and reports" / "IEEE_Results" / "_Hypothesis_Tests_IEEE.csv"
SIT_HYP_1 = PAPER_DIR / "figures and reports" / "SIT_Results" / "_Hypothesis_Tests_SIT_Set-1.csv"
SIT_HYP_2 = PAPER_DIR / "figures and reports" / "SIT_Results" / "_Hypothesis_Tests_SIT_Set-2.csv"

OUT_DIR = PAPER_DIR / "tables"


def _fmt3(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.3f}"


def _latex_escape_text(s: str) -> str:
    # Minimal escaping for table text
    return (
        str(s)
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_ieee_tables() -> None:
    # Streaming aggregation: mean + 95% CI for unseen-only rows.

    @dataclass
    class Agg:
        n: int = 0
        sum: float = 0.0
        sumsq: float = 0.0

        def add(self, x: float) -> None:
            self.n += 1
            self.sum += x
            self.sumsq += x * x

        def mean(self) -> float:
            return self.sum / self.n if self.n else float("nan")

        def sd(self) -> float:
            if self.n <= 1:
                return 0.0
            # Sample variance
            var = (self.sumsq - (self.sum * self.sum) / self.n) / (self.n - 1)
            return math.sqrt(max(var, 0.0))

        def ci95(self) -> float:
            if self.n <= 1:
                return 0.0
            return 1.96 * self.sd() / math.sqrt(self.n)

    # Keyed by (Algorithm, Attention)
    eval_agg: dict[tuple[str, str], Agg] = {}
    tool_agg: dict[tuple[str, str], Agg] = {}
    lam_agg: dict[tuple[str, str], Agg] = {}
    viol_agg: dict[tuple[str, str], Agg] = {}

    unseen_markers = {"N", "NO", "0", "FALSE"}

    with IEEE_EVALS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            self_eval = str(row.get("Self-eval", "")).strip().upper()
            if self_eval not in unseen_markers:
                continue

            algo = str(row.get("Algorithm", "")).strip() or "(missing)"
            attn = str(row.get("Attention Mechanism", "")).strip() or "(missing)"
            key = (algo, attn)

            def to_float(v: str) -> float | None:
                try:
                    if v is None:
                        return None
                    v = str(v).strip()
                    if v == "":
                        return None
                    return float(v)
                except Exception:
                    return None

            ev = to_float(row.get("Eval_Score"))
            tu = to_float(row.get("Tool Usage %"))
            la = to_float(row.get("Lambda"))
            vi = to_float(row.get("Threshold Violations"))

            if ev is not None:
                eval_agg.setdefault(key, Agg()).add(ev)
            if tu is not None:
                tool_agg.setdefault(key, Agg()).add(tu)
            if la is not None:
                lam_agg.setdefault(key, Agg()).add(la)
            if vi is not None:
                viol_agg.setdefault(key, Agg()).add(vi)

    rows: list[dict[str, object]] = []
    keys = set(eval_agg) | set(tool_agg) | set(lam_agg) | set(viol_agg)
    for algo, attn in keys:
        ea = eval_agg.get((algo, attn), Agg())
        ta = tool_agg.get((algo, attn), Agg())
        la = lam_agg.get((algo, attn), Agg())
        va = viol_agg.get((algo, attn), Agg())
        rows.append(
            {
                "Algorithm": algo,
                "Attention": attn,
                "n": ea.n,
                "EvalScore": ea.mean(),
                "EvalCI": ea.ci95(),
                "ToolUse": ta.mean(),
                "ToolCI": ta.ci95(),
                "Lambda": la.mean(),
                "LambdaCI": la.ci95(),
                "Viol": va.mean(),
                "ViolCI": va.ci95(),
            }
        )

    rows.sort(key=lambda r: (float(r["EvalScore"]), float(r["ToolUse"])), reverse=True)
    top = rows[:10]

    # IEEE unseen summary (top 10)
    lines = []
    lines.append("% Auto-generated from _Evals_Results_IEEE.csv (unseen-only)\n")
    lines.append("\\begin{tabular}{ll r r r r}\n")
    lines.append("\\toprule\n")
    lines.append(
        "\\rowcolor{gray!20} \\textbf{Algorithm} & \\textbf{Attention} & \\textbf{Eval. Score} & \\textbf{CI} & \\textbf{Tool use (\\%)} & \\textbf{$\\lambda$}\\\\\n"
    )
    lines.append("\\midrule\n")
    for r in top:
        lines.append(
            f"{_latex_escape_text(r['Algorithm'])} & { _latex_escape_text(r['Attention']) } & "
            f"{_fmt3(float(r['EvalScore']))} & {_fmt3(float(r['EvalCI']))} & {_fmt3(float(r['ToolUse']))} & {_fmt3(float(r['Lambda']))}\\\\\n"
        )
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    _write(OUT_DIR / "ieee_unseen_summary.tex", "".join(lines))

    # Best attention per algorithm
    # Best attention per algorithm (IEEE unseen-only)
    best_map: dict[str, dict[str, object]] = {}
    for r in rows:
        algo = str(r["Algorithm"])
        current = best_map.get(algo)
        if current is None or float(r["EvalScore"]) > float(current["EvalScore"]):
            best_map[algo] = r
    best = [best_map[k] for k in sorted(best_map.keys())]
    lines = []
    lines.append("% Auto-generated best attention per algorithm (IEEE unseen-only)\n")
    lines.append("\\begin{tabular}{l l r r r}\n")
    lines.append("\\toprule\n")
    lines.append(
        "\\rowcolor{gray!20} \\textbf{Algorithm} & \\textbf{Best attention} & \\textbf{Eval. Score} & \\textbf{CI} & \\textbf{n}\\\\\n"
    )
    lines.append("\\midrule\n")
    for r in best:
        lines.append(
            f"{_latex_escape_text(r['Algorithm'])} & { _latex_escape_text(r['Attention']) } & "
            f"{_fmt3(float(r['EvalScore']))} & {_fmt3(float(r['EvalCI']))} & {int(r['n'])}\\\\\n"
        )
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    _write(OUT_DIR / "ieee_best_attention_per_algo.tex", "".join(lines))


@dataclass(frozen=True)
class HypRow:
    dataset: str
    panel: str
    competitor: str
    metric: str
    t: str
    p: str
    sig: str
    wins: str
    n_reinforce: str
    n_other: str


def _load_hyp(path: Path, dataset: str) -> list[HypRow]:
    rows: list[HypRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                HypRow(
                    dataset=dataset,
                    panel=str(r.get("Panel", "")),
                    competitor=str(r.get("Competitor", "")),
                    metric=str(r.get("Metric", "")),
                    t=str(r.get("T-Statistic", "")),
                    p=str(r.get("P-Value (1-tail)", "")),
                    sig=str(r.get("Sig (α=0.05)", "")),
                    wins=str(r.get("REINFORCE wins?", "")),
                    n_reinforce=str(r.get("REINFORCE Sample Size", "")),
                    n_other=str(r.get("Others Sample Size", "")),
                )
            )
    return rows


def make_hypothesis_table() -> None:
    # Keep it compact: Eval Score only, unseen-only panel, for each dataset.
    all_rows: list[HypRow] = []
    all_rows += _load_hyp(IEEE_HYP, "IEEE")
    all_rows += _load_hyp(SIT_HYP_1, "SIT Set-1")
    all_rows += _load_hyp(SIT_HYP_2, "SIT Set-2")

    filtered = [
        r
        for r in all_rows
        if "Unseen Only" in r.panel and r.metric.strip().lower() == "eval score"
    ]

    # Order datasets then competitors
    dataset_order = {"SIT Set-1": 1, "SIT Set-2": 2, "IEEE": 3}
    competitor_order = {"A2C": 1, "DQN": 2, "PPO": 3}
    filtered.sort(key=lambda r: (dataset_order.get(r.dataset, 99), competitor_order.get(r.competitor, 99)))

    def _panel_short(panel: str) -> str:
        p = (panel or "").strip()
        if "All Attn" in p:
            return "All Attn"
        if "Best Attn" in p:
            return "Best Attn"
        return p

    def _to_float(s: str) -> float | None:
        try:
            if s is None:
                return None
            s = str(s).strip()
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    lines = []
    lines.append("% Auto-generated from *_Hypothesis_Tests_*.csv (Eval Score, Unseen Only)\n")
    lines.append("\\begin{tabular}{l l l r r l}\n")
    lines.append("\\toprule\n")
    lines.append(
        "\\rowcolor{gray!20} \\textbf{Dataset} & \\textbf{Panel} & \\textbf{Competitor} & \\textbf{t} & \\textbf{p (1-tail)} & \\textbf{Sig.}\\\\\n"
    )
    lines.append("\\midrule\n")
    for r in filtered:
        t_val = _to_float(r.t)
        p_val = _to_float(r.p)
        lines.append(
            f"{_latex_escape_text(r.dataset)} & {_latex_escape_text(_panel_short(r.panel))} & {_latex_escape_text(r.competitor)} & "
            f"{_fmt3(t_val) if t_val is not None else _latex_escape_text(r.t)} & "
            f"{_fmt3(p_val) if p_val is not None else _latex_escape_text(r.p)} & {_latex_escape_text(r.sig)}\\\\\n"
        )
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    _write(OUT_DIR / "hypothesis_tests_summary.tex", "".join(lines))


def main() -> None:
    missing = [p for p in (IEEE_EVALS, IEEE_HYP, SIT_HYP_1, SIT_HYP_2) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required input files: {missing}")

    make_ieee_tables()
    make_hypothesis_table()

    print("Wrote:")
    for p in sorted(OUT_DIR.glob("*.tex")):
        print(" -", p.relative_to(PAPER_DIR))


if __name__ == "__main__":
    main()
