#!/usr/bin/env python3
"""Valide un dataset LeRobot destiné au fine-tuning de SmolVLA et trace ses trajectoires.

Vérifie le format (fps, clés et formes attendues par lerobot/smolvla_base), la tâche unique,
le nombre et la durée des épisodes, les valeurs (NaN, articulations figées, sauts), l'écart
action (leader) / état (follower), et les images (décodage, image noire, flux figé).
Écrit dans --out : summary.txt, overview.png, lengths.png, trajectories_epNNN.png, frames.jpg.
Code de retour 1 si une vérification bloquante (FAIL) échoue.

  python3 tools/check_dataset.py --repo-id user/robotkraft_pickplace --task "Pick up ..." --out outputs/dataset_check/x
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
DARK = 20  # luminosité moyenne (0-255) d'une image jugée quasi noire


class Report:
    def __init__(self):
        self.lines, self.fails, self.warns = [], 0, 0

    def add(self, tag, msg):
        line = f"[{tag:4}] {msg}"
        print(line)
        self.lines.append(line)

    def ok(self, msg):
        self.add("OK", msg)

    def info(self, msg):
        self.add("", msg)

    def warn(self, msg):
        self.warns += 1
        self.add("WARN", msg)

    def fail(self, msg):
        self.fails += 1
        self.add("FAIL", msg)

    def check(self, cond, ok_msg, ko_msg, level="fail"):
        if cond:
            self.ok(ok_msg)
        else:
            getattr(self, level)(ko_msg)
        return cond


def check_meta(meta, args, rep):
    rep.info(f"dataset {meta.repo_id} : {meta.total_episodes} épisodes, {meta.total_frames} images, "
             f"robot_type={meta.robot_type}, version {meta.info.get('codebase_version')}")
    rep.check(meta.fps == args.fps, f"fps = {meta.fps}", f"fps = {meta.fps}, attendu {args.fps}")
    for key in ("observation.state", "action"):
        ft = meta.features.get(key)
        if ft is None:
            rep.fail(f"{key} absent")
            continue
        names = ft.get("names") or []
        rep.check(tuple(ft["shape"]) == (6,) and list(names) == JOINTS,
                  f"{key} : 6 dimensions {JOINTS}", f"{key} : forme {ft['shape']}, noms {names}, attendu 6 x {JOINTS}")
    expected = {f"observation.images.{c}" for c in args.cameras}
    found = set(meta.camera_keys)
    rep.check(found == expected, f"caméras {sorted(found)}",
              f"caméras {sorted(found)}, attendu {sorted(expected)} (clés de smolvla_base, sinon rename_map nécessaire)")
    for key in sorted(found):
        ft = meta.features[key]
        shape = tuple(ft["shape"])
        info = ft.get("info") or {}
        rep.check(shape == (args.height, args.width, 3), f"{key} : {shape} {ft['dtype']} {info.get('video.codec', '')}",
                  f"{key} : {shape}, attendu {(args.height, args.width, 3)}")
    tasks = list(meta.tasks.index)
    if args.task:
        rep.check(tasks == [args.task], f"tâche unique : {tasks[0]!r}" if tasks else "tâche",
                  f"tâches {tasks}, attendu exactement [{args.task!r}] (même phrase au record et au rollout)")
    else:
        rep.info(f"tâches : {tasks}")
    if args.expected_episodes:
        rep.check(meta.total_episodes >= args.expected_episodes, f"{meta.total_episodes} épisodes",
                  f"{meta.total_episodes} épisodes, objectif {args.expected_episodes}", level="warn")


def load_tables(root):
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"aucun parquet sous {root / 'data'}")
    cols = ["observation.state", "action", "episode_index", "frame_index", "index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


def check_values(df, fps, rep):
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy()
    rep.check(np.isfinite(state).all() and np.isfinite(action).all(), "aucune valeur NaN/inf", "valeurs NaN/inf présentes")

    lengths = df.groupby("episode_index").size()
    durations = lengths / fps
    rep.info(f"durée des épisodes : min {durations.min():.1f} s, médiane {durations.median():.1f} s, max {durations.max():.1f} s")
    short = durations[durations < 2.0]
    rep.check(short.empty, "aucun épisode de moins de 2 s",
              f"épisodes de moins de 2 s (raté ? à refaire) : {list(short.index)}", level="warn")
    contiguous = all((g["frame_index"].to_numpy() == np.arange(len(g))).all() for _, g in df.groupby("episode_index"))
    rep.check(contiguous, "frame_index contigu dans chaque épisode", "frame_index non contigu dans au moins un épisode")

    rep.info("amplitude par articulation (état follower) : " + ", ".join(
        f"{j.split('.')[0]} [{state[:, i].min():.0f}, {state[:, i].max():.0f}]" for i, j in enumerate(JOINTS)))
    still = [j for i, j in enumerate(JOINTS) if state[:, i].std() < 0.5]
    rep.check(not still, "toutes les articulations bougent", f"articulations quasi immobiles (moteur ? calibration ?) : {still}", level="warn")

    same_ep = ep[1:] == ep[:-1]
    jumps = np.abs(np.diff(state, axis=0))[same_ep]
    big = [JOINTS[i] for i in range(6) if jumps.size and jumps[:, i].max() > 20]
    rep.check(not big, "pas de saut d'état > 20 par image", f"sauts d'état > 20 par image (glitch de lecture ?) : {big}", level="warn")

    # Le follower suit l'action du leader avec une image de retard : écart moyen |action(t) - état(t+1)|.
    lag = np.abs(action[:-1] - state[1:])[same_ep]
    if lag.size:
        mean_lag = lag.mean(axis=0)
        rep.info("écart moyen action leader / état follower : " + ", ".join(
            f"{j.split('.')[0]} {m:.1f}" for j, m in zip(JOINTS, mean_lag)))
        off = [JOINTS[i] for i in range(5) if mean_lag[i] > 10]  # pince exclue : elle bute sur l'objet saisi
        rep.check(not off, "le follower suit le leader (écart < 10 hors pince)",
                  f"écart leader/follower > 10 (calibrations incohérentes ? bras gêné ?) : {off}", level="warn")
    no_grip = [int(e) for e, g in df.groupby("episode_index") if np.stack(g["observation.state"].to_numpy())[:, 5].std() < 1.0]
    rep.check(not no_grip, "la pince bouge dans chaque épisode", f"pince immobile dans les épisodes {no_grip}", level="warn")
    return state, action, ep, lengths


def plot_all(state, action, ep, lengths, fps, out, sample_eps):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for e in np.unique(ep):
        m = ep == e
        t = np.arange(m.sum()) / fps
        for i, ax in enumerate(axes.flat):
            ax.plot(t, state[m, i], lw=0.7, alpha=0.5)
    for i, ax in enumerate(axes.flat):
        ax.set_title(JOINTS[i])
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("temps (s)")
    fig.suptitle(f"État du follower, {len(lengths)} épisodes superposés (diversité des trajectoires)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(lengths / fps, bins=min(30, max(5, len(lengths))))
    ax.set_xlabel("durée de l'épisode (s)")
    ax.set_ylabel("épisodes")
    fig.tight_layout()
    fig.savefig(out / "lengths.png", dpi=110)
    plt.close(fig)

    for e in sample_eps:
        m = ep == e
        t = np.arange(m.sum()) / fps
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(t, state[m, i], label="état (follower)")
            ax.plot(t, action[m, i], "--", label="action (leader)")
            ax.set_title(JOINTS[i])
            ax.grid(alpha=0.3)
        axes[0, 0].legend()
        fig.suptitle(f"Épisode {e}")
        fig.tight_layout()
        fig.savefig(out / f"trajectories_ep{e:03d}.png", dpi=110)
        plt.close(fig)


def check_frames(repo_id, root, meta, df, sample_eps, out, rep):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    missing = [(e, k) for e in sample_eps for k in meta.video_keys
               if not (root / meta.get_video_file_path(int(e), k)).exists()]
    if missing:
        rep.warn(f"vidéos absentes en local ({len(missing)} fichiers) : images non vérifiées")
        return
    ds = LeRobotDataset(repo_id, root=root)
    dark = {k: [] for k in meta.camera_keys}
    for e, g in df.groupby("episode_index"):  # image du milieu de chaque épisode
        item = ds[int(g["index"].to_numpy()[len(g) // 2])]
        for k in meta.camera_keys:
            if float(item[k].mean()) * 255 < DARK:
                dark[k].append(int(e))
    for k, eps in dark.items():
        rep.check(not eps, f"{k} : aucune image noire (milieu de chaque épisode)",
                  f"{k} : image quasi noire au milieu des épisodes {eps} (objectif masqué ? éclairage ?)", level="warn")
    rows, labels = [], []
    for e in sample_eps:
        idx = df.loc[df["episode_index"] == e, "index"].to_numpy()
        for where, i in (("début", idx[0]), ("fin", idx[-1])):
            item = ds[int(i)]
            imgs = [(item[k].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8) for k in meta.camera_keys]
            rows.append(np.hstack(imgs))
            labels.append(f"ép. {e} {where}")
        first, last = rows[-2], rows[-1]
        w = first.shape[1] // len(meta.camera_keys)
        for c, k in enumerate(meta.camera_keys):
            diff = np.abs(first[:, c * w:(c + 1) * w].astype(int) - last[:, c * w:(c + 1) * w].astype(int)).mean()
            if diff < 1.0:
                rep.warn(f"{k}, épisode {e} : image identique au début et à la fin (flux figé ?)")
    rep.ok(f"images décodées : {len(rows)} instants x {len(meta.camera_keys)} caméras")
    fig, axes = plt.subplots(len(rows), 1, figsize=(4 * len(meta.camera_keys), 3 * len(rows)))
    for ax, img, label in zip(np.atleast_1d(axes), rows, labels):
        ax.imshow(img)
        ax.set_title(f"{label} : " + " | ".join(k.split(".")[-1] for k in meta.camera_keys), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "frames.jpg", dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None, help="dossier local (défaut : $HF_LEROBOT_HOME/<repo-id>)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cameras", nargs="+", default=["camera1", "camera2"])
    ap.add_argument("--task", default="", help="phrase attendue (TASK_TEXT)")
    ap.add_argument("--expected-episodes", type=int, default=0)
    ap.add_argument("--no-videos", action="store_true", help="ne pas décoder d'images")
    ap.add_argument("--out", default="outputs/dataset_check")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    root = Path(meta.root)
    check_meta(meta, args, rep)
    df = load_tables(root)
    state, action, ep, lengths = check_values(df, meta.fps, rep)
    episodes = sorted(np.unique(ep))
    sample_eps = sorted({episodes[0], episodes[len(episodes) // 2], episodes[-1]})
    plot_all(state, action, ep, lengths, meta.fps, out, sample_eps)
    if not args.no_videos:
        check_frames(args.repo_id, root, meta, df, sample_eps, out, rep)

    verdict = "ÉCHEC" if rep.fails else ("OK avec avertissements" if rep.warns else "OK")
    rep.info(f"bilan : {verdict} ({rep.fails} FAIL, {rep.warns} WARN). Graphiques : {out}/")
    (out / "summary.txt").write_text("\n".join(rep.lines) + "\n")
    sys.exit(1 if rep.fails else 0)


if __name__ == "__main__":
    main()
