# RobotKraft : SO-101 leader/follower -> dataset LeRobot -> fine-tuning SmolVLA -> rollout.
# Point d'entrée unique : `make` liste les cibles, phase par phase.
# Configuration : config/robotkraft.env (+ config/robotkraft.env.local, non versionné).
# Cibles [terminal] : interactives (Entrée, flèches, mouvements du bras), à lancer soi-même.
# Aucune cible n'appelle lerobot-setup-motors : les bras du challenge sont déjà configurés.

SHELL := /bin/bash
.DEFAULT_GOAL := help

include config/robotkraft.env
-include config/robotkraft.env.local

# ---------------------------------------------------------------------------
# Exécution. RUNNER=docker : conteneur NON privilégié avec uniquement les devices
# utiles. En mode privileged, Docker ignore un --device dont le chemin existe déjà
# sur l'hôte : /dev/ttyACM0 serait le premier bras branché, pas forcément le follower.
# ---------------------------------------------------------------------------
TTY_FLAGS := $(shell [ -t 0 ] && echo -it || echo -i)
GROUP_ADD := $(foreach g,dialout video plugdev,$(if $(shell getent group $(g)),--group-add $(shell getent group $(g) | cut -d: -f3)))
DOCKER_RUN = docker run --rm --init $(TTY_FLAGS) --user $(shell id -u):$(shell id -g) $(GROUP_ADD) \
	-e HOME=/workspace/.cache/home -e USER=$(shell id -un) -e LOGNAME=$(shell id -un) -v $(CURDIR):/workspace/robotkraft -w /workspace/robotkraft \
	-v $(abspath $(CACHE_DIR)):/workspace/.cache --shm-size=2g

export CAM_WIDTH CAM_HEIGHT CAM_FPS CAM1_ZMQ_PORT CAM1_ZMQ_NAME
ifeq ($(RUNNER),native)
export HF_HOME := $(abspath $(CACHE_DIR))/huggingface
export HF_LEROBOT_HOME := $(HF_HOME)/lerobot
run =
run_gpu =
else
run = $(DOCKER_RUN) $(1) $(IMAGE)
run_gpu = $(DOCKER_RUN) --gpus all $(1) $(TRAIN_IMAGE)
endif
TRAIN_IMAGE ?= robotkraft-smolvla:cuda
# Rollout : image CUDA + GPU si POLICY_DEVICE=cuda (machine NVIDIA), sinon image CPU.
run_policy = $(if $(filter cuda,$(POLICY_DEVICE)),$(call run_gpu,$(1)),$(call run,$(1)))

DEV_FOLLOWER = --device $(FOLLOWER_PORT):$(FOLLOWER_PORT)
DEV_LEADER   = --device $(LEADER_PORT):$(LEADER_PORT)
DEV_WRIST    = --device $(WRIST_CAM):$(WRIST_CAM)
# OAK-D : depthai redémarre le device USB (nouveau nœud) -> tout /dev/bus/usb + règle cgroup.
DEV_OAK      = -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw'
DEV_ALL_VIDEO = $(foreach d,$(wildcard /dev/video*),--device $(d):$(d)) $(if $(wildcard /dev/v4l/by-id),-v /dev/v4l:/host-v4l:ro)
# X11 : nécessaire aux touches de lerobot-record (→ ← Échap, via pynput).
DEV_X11      = -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix:ro $(if $(XAUTHORITY),-e XAUTHORITY=/tmp/.Xauthority -v $(XAUTHORITY):/tmp/.Xauthority:ro)
OAK_ENV      = -e CAM_WIDTH=$(CAM_WIDTH) -e CAM_HEIGHT=$(CAM_HEIGHT) -e CAM_FPS=$(CAM_FPS) -e CAM1_ZMQ_PORT=$(CAM1_ZMQ_PORT) -e CAM1_ZMQ_NAME=$(CAM1_ZMQ_NAME)

ROBOT_ARGS  = --robot.type=so101_follower --robot.port=$(FOLLOWER_PORT) --robot.id=$(FOLLOWER_ID)
TELEOP_ARGS = --teleop.type=so101_leader --teleop.port=$(LEADER_PORT) --teleop.id=$(LEADER_ID)
# Noms camera1/camera2 dès l'enregistrement = clés de lerobot/smolvla_base, sans rename_map.
CAMERAS = {camera1: {type: zmq, server_address: localhost, port: $(CAM1_ZMQ_PORT), camera_name: $(CAM1_ZMQ_NAME), width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS)}, camera2: {type: opencv, index_or_path: $(WRIST_CAM), fourcc: MJPG, width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS)}}

CALIB_DIR      = $(CACHE_DIR)/huggingface/lerobot/calibration
CALIB_FOLLOWER = $(CALIB_DIR)/robots/so_follower/$(FOLLOWER_ID).json
CALIB_LEADER   = $(CALIB_DIR)/teleoperators/so_leader/$(LEADER_ID).json
DATASET_DIR    = $(CACHE_DIR)/huggingface/lerobot/$(DATASET_REPO_ID)
EVAL_REPO_ID   = $(HF_USER)/eval_$(TASK_NAME)_$(shell date +%m%d_%H%M%S)
RUN       ?= $(TASK_NAME)
CONDITION ?= nominal

.PHONY: help install install-native install-cuda doctor ports ports-snapshot ports-diff cameras hf-login hf-whoami \
	udev import-calib calib-status calibrate-follower calibrate-leader teleop \
	record record-resume check-dataset push-dataset train-cmd train train-resume train-hf-job push-model \
	rollout policy-server rollout-async eval-log eval-report setup-motors prep \
	need-follower need-leader need-wrist need-oak need-hf-user need-image need-calib need-calib-follower

help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36mmake %-20s\033[0m %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))
	@echo ""
	@echo "Config : HF_USER=$(HF_USER)  DATASET=$(DATASET_REPO_ID)  RUNNER=$(RUNNER)"
	@echo "Tâche  : $(TASK_TEXT)"

##@ Phase 0 — Installation et détection (aucun moteur)
install: ## Vérifie Docker et les groupes, construit l'image (challenge + num2words)
	BASE_IMAGE=$(BASE_IMAGE) IMAGE=$(IMAGE) bash scripts/install.sh

install-cuda: ## Machine NVIDIA avec Docker : même chose sur l'image CUDA du challenge (~9 Go)
	TRAIN_IMAGE=$(TRAIN_IMAGE) bash scripts/install.sh cuda

install-native: ## Machine GPU sans Docker (serveur) : env conda + lerobot[feetech,smolvla]==0.5.1
	bash scripts/install.sh native

doctor: ## État du poste : image, bras, caméras, calibrations, compte HF
	@echo "== Image ($(RUNNER))"; [ "$(RUNNER)" != docker ] || docker image inspect $(IMAGE) --format '  OK $(IMAGE) (créée {{.Created}})' 2>/dev/null || echo "  ABSENTE $(IMAGE) : make install"
	@echo "== Bras"; for p in $(FOLLOWER_PORT) $(LEADER_PORT); do [ -e $$p ] && echo "  OK $$p -> $$(readlink -f $$p)" || echo "  ABSENT $$p"; done
	@echo "== Caméras"; [ -e $(WRIST_CAM) ] && echo "  OK camera2 (poignet) -> $$(readlink -f $(WRIST_CAM))" || echo "  ABSENTE camera2 (poignet) $(WRIST_CAM)"; \
		grep -qs 03e7 /sys/bus/usb/devices/*/idVendor && echo "  OK camera1 (OAK-D, USB 03e7)" || echo "  ABSENTE camera1 (OAK-D)"
	@echo "== Calibrations"; for f in $(CALIB_FOLLOWER) $(CALIB_LEADER); do [ -s $$f ] && echo "  OK $$f" || echo "  ABSENTE $$f"; done
	@echo "== Hugging Face"; [ -s $(CACHE_DIR)/huggingface/token ] && echo "  token présent (make hf-whoami)" || echo "  pas de token : make hf-login"; \
		[ "$(HF_USER)" != CHANGE_ME ] && echo "  HF_USER=$(HF_USER)" || echo "  HF_USER non renseigné (config/robotkraft.env.local)"
	@echo "== Dataset"; [ -d $(DATASET_DIR) ] && echo "  $(DATASET_DIR) présent" || echo "  $(DATASET_DIR) pas encore enregistré"

ports: ## Liste les ports série USB (VID/PID, serial, chemin USB)
	@python3 scripts/detect_ports.py

ports-snapshot: ## Mémorise les ports présents, avant de brancher ou débrancher un bras
	@python3 scripts/detect_ports.py snapshot

ports-diff: ## Montre le port apparu ou disparu depuis ports-snapshot
	@python3 scripts/detect_ports.py diff

cameras: need-image prep ## Liste les caméras, capture une image de chacune dans outputs/cam_check/
	$(call run,$(DEV_ALL_VIDEO) $(DEV_OAK)) python3 scripts/detect_cameras.py \
		--width $(CAM_WIDTH) --height $(CAM_HEIGHT) --fps $(CAM_FPS) --wrist $(WRIST_CAM) --out outputs/cam_check

hf-login: need-image prep ## [terminal] Connexion Hugging Face (token dans .cache/, jamais dans le repo)
	$(call run,) hf auth login

hf-whoami: need-image prep ## Affiche le compte Hugging Face utilisé
	$(call run,) hf auth whoami

##@ Phase 1 — Ports stables, calibration, téléopération
udev: ## [terminal] Crée /dev/so101_follower et /dev/so101_leader (sudo après confirmation)
	bash scripts/udev_rules.sh

import-calib: ## Copie les calibrations du repo du challenge (SRC=~/robotkraft-participant/.cache)
	@src=$(or $(SRC),$(HOME)/robotkraft-participant/.cache)/huggingface/lerobot/calibration; \
	for rel in robots/so_follower/$(FOLLOWER_ID).json teleoperators/so_leader/$(LEADER_ID).json; do \
		if [ -s $$src/$$rel ]; then mkdir -p $(CALIB_DIR)/$$(dirname $$rel) && cp -v $$src/$$rel $(CALIB_DIR)/$$rel; \
		else echo "absent : $$src/$$rel"; fi; done

calib-status: ## Montre les calibrations présentes
	@for f in $(CALIB_FOLLOWER) $(CALIB_LEADER); do [ -s $$f ] && echo "OK $$f ($$(date -r $$f '+%d/%m %H:%M'))" || echo "ABSENTE $$f"; done

calibrate-follower: need-follower need-image prep ## [terminal] lerobot-calibrate du follower (une fois)
	$(call run,$(DEV_FOLLOWER)) lerobot-calibrate $(ROBOT_ARGS)

calibrate-leader: need-leader need-image prep ## [terminal] lerobot-calibrate du leader (une fois)
	$(call run,$(DEV_LEADER)) lerobot-calibrate $(TELEOP_ARGS)

teleop: need-follower need-leader need-calib need-image prep ## [terminal] Le follower suit le leader (Ctrl+C pour arrêter)
	$(call run,$(DEV_FOLLOWER) $(DEV_LEADER)) lerobot-teleoperate $(ROBOT_ARGS) $(TELEOP_ARGS)

# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------
setup-motors:
	@echo "REFUSÉ : lerobot-setup-motors réécrit l'EEPROM des moteurs (ids, baudrate)."
	@echo "Les bras du challenge sont déjà configurés. Servo muet ou mal numéroté : va voir un coach."
	@exit 1

prep:
	@mkdir -p $(CACHE_DIR)/home outputs

need-follower:
	@test -e $(FOLLOWER_PORT) || { echo "ERREUR : $(FOLLOWER_PORT) absent. Follower branché (USB + alim 12 V) ? Règles udev posées (make udev) ?"; exit 1; }
need-leader:
	@test -e $(LEADER_PORT) || { echo "ERREUR : $(LEADER_PORT) absent. Leader branché (USB + alim 5 V) ? Règles udev posées (make udev) ?"; exit 1; }
need-wrist:
	@test -e $(WRIST_CAM) || { echo "ERREUR : caméra poignet $(WRIST_CAM) absente. make cameras, puis corrige WRIST_CAM."; exit 1; }
need-oak:
	@grep -qs 03e7 /sys/bus/usb/devices/*/idVendor || { echo "ERREUR : OAK-D (USB 03e7) non détectée. Branche-la, de préférence sur un port USB 3."; exit 1; }
need-hf-user:
	@[ "$(HF_USER)" != CHANGE_ME ] || { echo "ERREUR : HF_USER non renseigné. Exemple : echo HF_USER=monpseudo >> config/robotkraft.env.local"; exit 1; }
need-image:
	@[ "$(RUNNER)" != docker ] || docker image inspect $(IMAGE) >/dev/null 2>&1 || { echo "ERREUR : image $(IMAGE) absente. Lance make install."; exit 1; }
need-calib-follower:
	@[ -s $(CALIB_FOLLOWER) ] || { echo "ERREUR : calibration absente : $(CALIB_FOLLOWER). make calibrate-follower (ou make import-calib)."; exit 1; }
need-calib: need-calib-follower
	@[ -s $(CALIB_LEADER) ] || { echo "ERREUR : calibration absente : $(CALIB_LEADER). make calibrate-leader (ou make import-calib)."; exit 1; }
