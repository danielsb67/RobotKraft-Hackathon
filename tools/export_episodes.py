#!/usr/bin/env python3
"""Exporte chaque épisode d'un dataset LeRobot en vidéo H.264 lisible partout, caméras côte à côte.

LeRobot v3.0 met les épisodes bout à bout dans quelques gros fichiers AV1 : ce script les découpe
(ep_000.mp4, ep_001.mp4...) pour pouvoir les regarder et trier avant make push-dataset.
Le dataset n'est pas modifié.

  python3 tools/export_episodes.py --repo-id dbal67/robotkraft_quick --root <dossier> --out outputs/dataset_view/x
"""
import argparse
import subprocess
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo-id", required=True)
ap.add_argument("--root", required=True, help="dossier local du dataset (meta/, data/, videos/)")
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", nargs="*", type=int, help="seulement ces épisodes (défaut : tous)")
args = ap.parse_args()

meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
keys = meta.video_keys
for f in out.glob("ep_*.mp4"):  # les numéros changent après une suppression : on repart de zéro
    if args.episodes is None:
        f.unlink()

for ep in meta.episodes:
    e = ep["episode_index"]
    if args.episodes and e not in args.episodes:
        continue
    inputs = []
    for k in keys:
        path = root / meta.video_path.format(video_key=k, chunk_index=ep[f"videos/{k}/chunk_index"],
                                             file_index=ep[f"videos/{k}/file_index"])
        start, end = ep[f"videos/{k}/from_timestamp"], ep[f"videos/{k}/to_timestamp"]
        inputs += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(path)]
    stack = "".join(f"[{i}:v]" for i in range(len(keys))) + f"hstack=inputs={len(keys)}" if len(keys) > 1 else "null"
    dst = out / f"ep_{e:03d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", stack,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(dst)], check=True)
    print(f"ép. {e:3d} : {ep['length'] / meta.fps:5.1f} s  {ep['tasks'][0]!r}  -> {dst}")

print(f"\n{meta.total_episodes} épisodes ; caméras de gauche à droite : {', '.join(k.split('.')[-1] for k in keys)}")
print(f"Ouvrir le dossier : xdg-open {out}")
print('Supprimer les mauvais : make delete-episodes EPISODES="3 7" (numéros des fichiers ep_003, ep_007)')
