#!/usr/bin/env python3
"""Client d'inférence asynchrone officiel de LeRobot, avec la caméra ZMQ (camera1 = OAK-D) enregistrée.

En 0.5.1, lerobot.async_inference.robot_client n'importe que les configs caméra OpenCV et RealSense :
`type: zmq` y est refusé. Même contournement que scripts/robot/robot_client_zmq.py du repo du challenge.
"""
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401  (enregistre type: zmq)
from lerobot.async_inference.robot_client import async_client, register_third_party_plugins

if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
