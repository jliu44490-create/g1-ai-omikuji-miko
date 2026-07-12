#!/usr/bin/env python3
"""
g1_camera_receiver.py
Standard ZMQ receiver for Unitree G1 RealSense streams.

Compatible with teleimager ZMQ output:
- JPEG compressed stream (most common)
- Raw numpy (BGR, HWC)
- Multipart: JSON metadata + image bytes

Author: Unitree G1 teleop practice
"""

import zmq
import cv2
import numpy as np
import time


class G1CameraReceiver:
    def __init__(self, ip: str, port: int = 55555, topic: str = ""):
        """
        Args:
            ip: G1 IP address
            port: ZMQ port (default 55555)
            topic: (optional) pub/sub topic prefix
        """
        self.ip = ip
        self.port = port
        self.topic = topic.encode("utf-8") if topic else b""

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{self.ip}:{self.port}")
        self.sock.setsockopt(zmq.SUBSCRIBE, self.topic)

        self.last_meta = None
        self._frame_count = 0
        self._last_time = time.time()

        print(f"[G1CameraReceiver] Connected to tcp://{self.ip}:{self.port}")

    def _try_decode_jpeg(self, buf: bytes):
        img = cv2.imdecode(
            np.frombuffer(buf, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        return img

    def _try_decode_raw(self, buf: bytes, shape=(480, 640, 3)):
        try:
            return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        except Exception:
            return None

    def recv(self, timeout_ms=1000):
        """
        Receive one image frame.

        Returns:
            img (np.ndarray): BGR image, or None
            meta (dict): metadata if available, else None
        """
        if self.sock.poll(timeout_ms) == 0:
            return None, None

        # Try multipart (JSON + image)
        try:
            parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None, None

        # Case 1: multipart (meta + image)
        if len(parts) == 2:
            meta_bytes, img_bytes = parts
            try:
                import json
                meta = json.loads(meta_bytes.decode("utf-8"))
                self.last_meta = meta
            except Exception:
                meta = self.last_meta

            img = self._try_decode_jpeg(img_bytes)
            if img is None:
                shape = meta.get("shape", [480, 640, 3]) if meta else [480, 640, 3]
                img = self._try_decode_raw(img_bytes, tuple(shape))
            return img, meta

        # Case 2: single part (likely JPEG)
        data = parts[0]
        img = self._try_decode_jpeg(data)
        if img is not None:
            return img, self.last_meta

        # Case 3: raw numpy (fallback)
        img = self._try_decode_raw(data)
        return img, self.last_meta

    def spin_once(self, window_name="head_camera"):
        img, meta = self.recv()
        if img is None:
            return False

        self._frame_count += 1
        now = time.time()
        if now - self._last_time >= 1.0:
            fps = self._frame_count / (now - self._last_time)
            self._frame_count = 0
            self._last_time = now
            if meta is None:
                meta = {}
            meta["fps"] = f"{fps:.1f}"

        if meta and "fps" in meta:
            cv2.putText(
                img,
                f"FPS: {meta['fps']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, img)
        return True

    def run(self, window_name="head_camera"):
        print("[G1CameraReceiver] Running... Press 'q' to exit.")
        try:
            while True:
                if not self.spin_once(window_name):
                    time.sleep(0.001)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.close()

    def close(self):
        self.sock.close()
        self.ctx.term()
        cv2.destroyAllWindows()
        print("[G1CameraReceiver] Closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1 Camera ZMQ Receiver")
    parser.add_argument("--ip", required=True, help="G1 IP address")
    parser.add_argument("--port", type=int, default=55555, help="ZMQ port")
    args = parser.parse_args()

    receiver = G1CameraReceiver(args.ip, args.port)
    receiver.run()