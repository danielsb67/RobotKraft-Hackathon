# RobotKraft : SO-101 → dataset LeRobot → fine-tuning SmolVLA → rollout

On téléopère le bras follower avec le leader et on enregistre ~50 épisodes d'une tâche de pick-and-place,
vus par deux caméras. On fine-tune `lerobot/smolvla_base` sur ces épisodes, puis le modèle pilote seul
le follower et on mesure son taux de succès.

Tout passe par `make`, qui appelle les commandes officielles `lerobot-*` (LeRobot 0.5.1, celle de l'image
Docker du challenge). Pas de boucle de contrôle maison, pas de modification de LeRobot.
Toute la configuration est dans [config/robotkraft.env](config/robotkraft.env). `make` liste les cibles.

> Lance les commandes interactives (`udev`, `calibrate-*`, `teleop`, `record`, `rollout`, `eval-log`)
> depuis le **terminal intégré de VS Code**. Les touches de `lerobot-record` (→ ← Échap) passent par X11 :
> VS Code est un client X11 et les reçoit, le Terminal GNOME (Wayland) probablement pas.

## Démarrage rapide : du PC éteint au premier épisode

| # | Commande | Ce qu'il faut voir |
|---|---|---|
| 1 | `cd ~/robotkraft && make doctor` | Image OK, caméras OK. Bras et calibrations « ABSENT » au premier passage. |
| 2 | `echo "HF_USER=ton_pseudo_hf" > config/robotkraft.env.local` | Puis édite `TASK_NAME` et `TASK_TEXT` dans `config/robotkraft.env`. |
| 3 | `make hf-login`, puis `make hf-whoami` | Colle un token **write** (huggingface.co/settings/tokens). Ton pseudo s'affiche. |
| 4 | Brancher : follower (alim **12 V**), leader (alim **5 V**), OAK-D (USB 3), caméra poignet | |
| 5 | `make udev` (une seule fois) | Suis les questions. sudo est demandé à la fin, après affichage de la règle. `OK /dev/so101_follower -> ...` |
| 6 | `make cameras` | Ouvre `outputs/cam_check/all.jpg` : camera1 voit tout l'espace de travail, camera2 voit la pince, net. |
| 7 | `make calibrate-follower`, puis `make calibrate-leader` | Fichiers `.json` créés (`make calib-status`). |
| 8 | `make teleop` | Le follower copie le leader. Ctrl+C pour arrêter. |
| 9 | `make record NUM_EPISODES=3` | 3 épisodes d'essai. Ensuite `make check-dataset`. |
| 10 | Si le contrôle est OK : `make record-resume NUM_EPISODES=47` | Le dataset atteint 50 épisodes. |
| 11 | `make push-dataset`, puis entraînement (Phase 3) | |

Le détail de chaque phase, les vérifications physiques et les erreurs courantes sont plus bas.

## Comment c'est construit

```
leader (5 V) ──USB──> /dev/so101_leader ──┐
follower (12 V) ─USB─> /dev/so101_follower ┼─> lerobot-record ──> dataset LeRobot v3.0
OAK-D Lite ─USB─> oak_zmq_server.py ─ZMQ─> camera1 (vue fixe)      (.cache/huggingface/lerobot/<HF_USER>/robotkraft_<TASK_NAME>)
webcam poignet ─USB─> OpenCV ──────────> camera2 (poignet)              │
                                                                        ▼ make push-dataset (Hub, privé)
                     serveur GPU : lerobot-train --policy.path=lerobot/smolvla_base
                                                                        │ modèle sur le Hub
                                                                        ▼
                     PC : lerobot-record --policy.path=...  (le modèle pilote le follower)
```

Choix et raisons (sources et vérifications : [docs/REFERENCES.md](docs/REFERENCES.md)) :

- **Clés `camera1` / `camera2` dès l'enregistrement.** Ce sont les noms d'entrée de `smolvla_base` : pas de
  rename_map, et SmolVLA 0.5.1 ignore la camera3 absente. Avec d'autres noms (`top`, `wrist`...), le chargement
  de la policy échoue sur « Feature mismatch ».
- **État = 6 positions du follower, action = 6 positions du leader.** C'est ce que produit `lerobot-record` avec
  un SO-101 : `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- **640x480 à 30 fps, profondeur coupée.** C'est le format du dataset de référence de SmolVLA (svla_so101_pickplace).
  La caméra ZMQ de LeRobot ne redimensionne pas : le pont OAK (`scripts/oak_zmq_server.py`) produit directement
  du 640x480. Ce format couvre tout le champ du capteur 4:3 de l'OAK-D Lite.
- **Docker non privilégié.** L'image du challenge (LeRobot 0.5.1) reçoit en plus `num2words`, sans lequel SmolVLA
  ne se charge pas ([docker/Dockerfile](docker/Dockerfile)). Les conteneurs ne reçoivent que les devices utiles.
  En mode `privileged` (repo du challenge), Docker ignore le mapping `/dev/lerobot_follower:/dev/ttyACM0` :
  le « follower » y est en fait le premier bras branché.
- **Calibrations, datasets, modèles et token HF** sont dans `.cache/`, ignoré par git, comme `outputs/`.
- **Mêmes ids que le repo du challenge** (`follower_arm`, `leader_arm`) : `make import-calib` reprend ses calibrations.

## Phase 0 — Installation et détection

```bash
make install    # déjà fait sur ce PC ; ~1 min si l'image du challenge est présente
make doctor     # état général
make ports      # ports série USB, avec leur chemin USB physique
make cameras    # une image par caméra dans outputs/cam_check/, débit, alerte image noire
```

À vérifier physiquement :
- camera1 (OAK-D) voit tout l'espace de travail : le bras, la zone de prise, la zone de dépôt. Mieux vaut la
  surélever et l'incliner vers la table qu'une vue de profil avec beaucoup de mur.
- camera2 (poignet) voit les mors de la pince et l'objet, net. Règle la mise au point à la main (bague de l'objectif)
  sur environ 10 à 20 cm.
- `make cameras` ne doit signaler aucune « image quasi noire » quand la salle est éclairée.

## Phase 1 — Ports stables, calibration, téléopération

```bash
make udev                 # identifie les bras en les branchant un par un, crée /dev/so101_*
make calibrate-follower   # [terminal]
make calibrate-leader     # [terminal]
make teleop               # [terminal]
```

- **`make udev`** demande de débrancher le USB des deux bras, de brancher le follower seul, puis le leader.
  Si les deux cartes ont des numéros de série différents, la règle s'appuie dessus. Sinon, elle s'appuie sur le
  port USB physique : étiquette alors les ports et rebranche toujours chaque bras au même endroit.
- **Calibration** (`lerobot-calibrate`, une fois par bras) : place le bras au milieu de sa course, Entrée.
  Ensuite, bouge chaque articulation d'une butée à l'autre, une par une, pince comprise. Entrée pour finir.
  Si les calibrations existent déjà dans le repo du challenge : `make import-calib`.
- **Si LeRobot demande « Press ENTER to use provided calibration file... or type 'c' »** : les moteurs ont été
  calibrés ailleurs (autre PC). Entrée réécrit dans les moteurs la calibration de ce PC. `c` refait une calibration.
- **Téléopération** : avant de lancer, mets le leader dans la même pose que le follower. Au démarrage, le follower
  rejoint d'un coup la pose du leader. Garde une main près de l'alimentation 12 V du follower.

Ne lance jamais `lerobot-setup-motors` (`make setup-motors` refuse exprès) : il réécrit l'EEPROM des moteurs, et
les bras du challenge sont déjà configurés.

## Phase 2 — Enregistrement du dataset

```bash
make record NUM_EPISODES=3          # essai court : vérifie tout avant d'enregistrer en série
make check-dataset                   # rapport + graphiques dans outputs/dataset_check/<dataset>/
make record-resume NUM_EPISODES=47   # complète jusqu'à 50
make push-dataset                    # envoie sur le Hub (privé par défaut)
```

Pendant `lerobot-record` :
- chaque épisode dure au plus `EPISODE_TIME_S` (20 s), puis vient une pause de remise en place de `RESET_TIME_S` (10 s) ;
- **→** termine l'épisode tout de suite (tâche réussie avant la fin), **←** le jette et le refait, **Échap** arrête
  proprement (le dataset reste valide) ;
- garde la même phrase de tâche (`TASK_TEXT`) pour tout le dataset et pour le rollout.

Pour que le modèle généralise (c'est ce que le jury évalue) :
- varie la position de départ de l'objet sur toute la zone : environ 10 épisodes par zone ou variation ;
- varie un peu l'éclairage et ajoute quelques distracteurs dans une partie des épisodes ;
- fais des gestes nets et réguliers, sans hésitation : le modèle copie aussi les hésitations ;
- jette (←) les épisodes ratés plutôt que de les garder.

`make check-dataset` signale en FAIL : mauvais fps, clés ou formes, plusieurs phrases de tâche, NaN.
Il signale en WARN : épisodes trop courts, articulation ou pince immobile, sauts d'état, écart leader/follower,
images noires ou figées. Regarde `overview.png` (diversité des trajectoires) et `frames.jpg` (ce que voit le modèle).

## Phase 3 — Fine-tuning SmolVLA

Le PC n'a pas de GPU NVIDIA. Trois options :

1. **Serveur GPU du challenge (gratuit, recommandé)**. `make train-cmd` affiche les 3 commandes à coller dans un
   terminal JupyterLab du serveur : installation de `lerobot[smolvla]==0.5.1` si absent, `hf auth login`, puis
   l'entraînement en arrière-plan. Le serveur télécharge le dataset depuis le Hub et y renvoie le modèle à la fin.
2. **Une machine NVIDIA avec Docker** (ta tour 5070 Ti, par exemple) : copie ce dossier,
   puis `make install-cuda && make train`.
3. **HF Jobs (payant, facturé à la seconde)** : `make train-hf-job` affiche la commande et le coût sans rien lancer.
   Pour lancer réellement : `make train-hf-job CONFIRM_PAID=yes`.

Réglages (`config/robotkraft.env`) :
- `TRAIN_STEPS=20000` : la doc SmolVLA compte environ 4 h sur un A100 complet.
- `BATCH_SIZE=32` : 64 dans la doc (sur A100). À ajuster selon la mémoire GPU, et à baisser en cas de « CUDA out of memory ».
- `SAVE_FREQ=5000` : un checkpoint tous les 5000 pas. `make train-resume` reprend au dernier.

Surveille la loss dans le log : elle doit baisser nettement pendant les premiers milliers de pas.

## Phase 4 — Rollout et taux de succès

```bash
make rollout POLICY_PATH=ton_pseudo/smolvla_robotkraft_pickplace   # modèle sur le Hub, ou un dossier local
make eval-log RUN=v1 CONDITION=nominal                              # 2e terminal : s / p / e après chaque essai
make eval-report                                                    # tableau + graphique dans outputs/eval/
```

- `make rollout` lance `lerobot-record --policy.path=...`. Le modèle pilote le follower pendant `EVAL_EPISODES`
  essais de `EVAL_EPISODE_TIME_S`, avec une pause de remise en place entre deux. Chaque essai est enregistré
  dans un dataset `eval_...` daté.
- **L'inférence se fait sur le CPU du PC** (`POLICY_DEVICE=cpu`) : le bras marque une pause à chaque nouveau paquet
  de 50 actions. Si c'est trop lent, fais tourner le modèle sur une machine GPU du même réseau :
  - sur la machine GPU : `make policy-server` ;
  - sur le PC : `make rollout-async ASYNC_SERVER=ip_de_la_machine:8080`.
- Conditions à tester (critères du jury) : `nominal`, `position`, `eclairage`, `distracteurs`, `objet_nouveau`.
  Fais 10 essais par condition. L'intervalle de confiance du rapport rappelle qu'avec 10 essais, l'écart entre deux
  versions doit être grand pour être significatif.

## Dépannage

| Symptôme | Cause probable et solution |
|---|---|
| `ERREUR : /dev/so101_follower absent` | Bras débranché, ou `make udev` pas fait. Avec la règle par port USB : bras rebranché sur un autre port. |
| `make teleop` : le follower ne bouge pas, ou bouge à l'envers | Bras inversés dans la règle udev : refais `make udev`. Sinon calibration ratée : `make calibrate-follower`. |
| `FileExistsError` ou `existe déjà` au record | Le dataset existe : `make record-resume`, ou change `TASK_NAME`. |
| Les touches → ← Échap ne font rien | Terminal Wayland : relance depuis le terminal de VS Code. Sinon, ouvre une session « Ubuntu sur Xorg ». |
| `aucune image de l'OAK-D` | Rebranche l'OAK-D sur un port USB 3 et ferme les autres programmes qui l'utilisent. |
| `Record loop is running slower` | PC surchargé : ferme les autres programmes. L'OAK doit être sur un port USB 3. |
| `Feature mismatch` au chargement de la policy | Noms de caméras différents de `camera1` / `camera2`. |
| `'policy.repo_id' argument missing` | Entraînement avec `PUSH_MODEL=true` sans `MODEL_REPO_ID` : garde la config par défaut. |
| `CUDA out of memory` | Baisse `BATCH_SIZE`, par exemple `make train BATCH_SIZE=16`. |

## Fichiers

```
Makefile                 point d'entrée unique (make = aide)
config/robotkraft.env    ports, ids, caméras, tâche, HF_USER, dataset, durées, entraînement
docker/Dockerfile        image du challenge + num2words (extra smolvla)
scripts/install.sh       installation : docker (défaut), cuda, native (conda Miniforge)
scripts/detect_ports.py  ports série USB, modes snapshot/diff
scripts/detect_cameras.py caméras + une capture chacune dans outputs/cam_check/
scripts/udev_rules.sh    symlinks /dev/so101_follower et /dev/so101_leader
scripts/oak_zmq_server.py pont OAK-D -> ZMQ (camera1), repris du repo du challenge, résolution paramétrable
scripts/with_oak.sh      lance le pont OAK le temps d'une commande
tools/check_dataset.py   validation du dataset + graphiques
tools/eval_logger.py     journal des essais -> CSV, tableau, graphique
docs/REFERENCES.md       sources et décisions
```
