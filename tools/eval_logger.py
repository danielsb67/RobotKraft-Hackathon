#!/usr/bin/env python3
"""Journal des essais de rollout -> CSV, tableau des taux de succès, graphique.

  python3 tools/eval_logger.py log --run v1 --condition nominal [--policy PATH] [--task TEXTE]
      Une ligne par essai : s = succès, p = partiel (0,5), e = échec, u = annuler le dernier, q = quitter.
      Une note libre peut suivre la lettre : "e a lâché le cube".
  python3 tools/eval_logger.py report [--csv outputs/eval/trials.csv]
      Taux par run et condition avec intervalle de Wilson à 95 %, écrit report.md et report.png.

Chaque essai est écrit tout de suite dans le CSV : rien n'est perdu si le terminal se ferme.
Conditions conseillées (critères du challenge) : nominal, position, eclairage, distracteurs, objet_nouveau.
"""
import argparse
import csv
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime

DEFAULT_CSV = os.path.join("outputs", "eval", "trials.csv")
FIELDS = ["timestamp", "run", "condition", "trial", "outcome", "score", "note", "policy", "task"]
OUTCOMES = {"s": ("succès", 1.0), "p": ("partiel", 0.5), "e": ("échec", 0.0)}


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson pour une proportion k/n (k peut contenir des demi-succès)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cmd_log(args):
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    rows = read_rows(args.csv)
    trial = 1 + max((int(r["trial"]) for r in rows if r["run"] == args.run and r["condition"] == args.condition), default=0)
    print(f"Run {args.run!r}, condition {args.condition!r}. Essai n°{trial}.")
    print("s = succès, p = partiel, e = échec (+ note facultative), u = annuler le dernier, q = quitter")
    added = 0
    while True:
        try:
            line = input(f"essai {trial} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        key, _, note = line.partition(" ")
        key = key.lower()
        if key == "q":
            break
        if key == "u":
            if added == 0:
                print("  rien à annuler dans cette session")
                continue
            rows = read_rows(args.csv)
            removed = rows.pop()
            write_rows(args.csv, rows)
            added -= 1
            trial -= 1
            print(f"  annulé : essai {removed['trial']} ({removed['outcome']})")
            continue
        if key not in OUTCOMES:
            print("  réponse inconnue : s, p, e, u ou q")
            continue
        outcome, score = OUTCOMES[key]
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "run": args.run, "condition": args.condition,
               "trial": trial, "outcome": outcome, "score": score, "note": note.strip(), "policy": args.policy, "task": args.task}
        new_file = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        added += 1
        trial += 1
        scores = [float(r["score"]) for r in read_rows(args.csv) if r["run"] == args.run and r["condition"] == args.condition]
        print(f"  {outcome}. Taux actuel : {sum(scores):g}/{len(scores)} = {100 * sum(scores) / len(scores):.0f} %")
    print(f"{added} essai(s) ajouté(s) dans {args.csv}. Bilan : make eval-report")


def cmd_report(args):
    rows = read_rows(args.csv)
    if not rows:
        sys.exit(f"Aucun essai dans {args.csv} : commence par make eval-log")
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["run"], r["condition"]), []).append(float(r["score"]))
    lines = ["| run | condition | essais | succès | taux | IC 95 % |", "|---|---|---|---|---|---|"]
    stats = []
    for (run, cond), scores in groups.items():
        n, k = len(scores), sum(scores)
        lo, hi = wilson(k, n)
        stats.append((f"{run} / {cond}", k / n, lo, hi, n))
        lines.append(f"| {run} | {cond} | {n} | {k:g} | {100 * k / n:.0f} % | {100 * lo:.0f}-{100 * hi:.0f} % |")
    table = "\n".join(lines)
    print(table)
    out_dir = os.path.dirname(args.csv) or "."
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(f"# Taux de succès ({datetime.now():%d/%m %H:%M})\n\n{table}\n\n"
                "Partiel = 0,5. Intervalle de Wilson à 95 % : avec 10 essais, il reste large.\n")
    try:
        import warnings

        warnings.filterwarnings("ignore", message="Unable to import Axes3D")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : pas de graphique (report.md écrit)")
        return
    labels = [s[0] for s in stats]
    rates = [100 * s[1] for s in stats]
    err = [[100 * (s[1] - s[2]) for s in stats], [100 * (s[3] - s[1]) for s in stats]]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(stats) + 1.5))
    ax.barh(labels, rates, xerr=err, capsize=4, color="#4C72B0")
    for y, s in enumerate(stats):
        ax.text(min(100 * s[3] + 2, 101), y, f"{100 * s[1]:.0f} % (n={s[4]})", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("taux de succès (%), barres = IC 95 % de Wilson")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "report.png"), dpi=120)
    print(f"Écrit : {out_dir}/report.md, {out_dir}/report.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    log = sub.add_parser("log", help="noter des essais")
    log.add_argument("--run", required=True, help="nom de la version testée, ex. v1_20k")
    log.add_argument("--condition", default="nominal")
    log.add_argument("--policy", default="")
    log.add_argument("--task", default="")
    log.add_argument("--csv", default=DEFAULT_CSV)
    rep = sub.add_parser("report", help="tableau et graphique")
    rep.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()
    cmd_log(args) if args.cmd == "log" else cmd_report(args)


if __name__ == "__main__":
    main()
