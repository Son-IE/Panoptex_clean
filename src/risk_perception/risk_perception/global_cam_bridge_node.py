#!/usr/bin/env python3
"""
global_cam_bridge_node.py  —  TRACK 1a  (TCP version, pairs with stream_sender_tcp.py)

Speaks the TCP counterpart of the old UDP protocol:

    packet = struct.pack(">L", len(jpeg)) + jpeg      # sent via sendall()

Unlike UDP, TCP has no message boundaries -- a single recv() may return a
partial frame, multiple frames, or a frame split across calls. So instead of
one recvfrom() == one datagram, we bind+listen+accept, then loop reading
exactly HEADER.size bytes for the length, then exactly that many bytes for
the JPEG payload, before decoding. No frame-size ceiling anymore (no 65507
byte datagram limit), and a single lost segment no longer costs a whole
frame -- TCP retransmits transparently.

Publishes:
    /global_cam/image_raw             (sensor_msgs/Image, bgr8)
    /global_cam/image_raw/compressed  (sensor_msgs/CompressedImage, jpeg passthrough)
    /global_cam/camera_info           (sensor_msgs/CameraInfo)

RECEIVE vs. PUBLISH ARE DECOUPLED ON PURPOSE. The receive thread's only job is
to drain the socket into a single-slot buffer -- no decode, no `publish()` --
so a slow or absent DDS subscriber can never block it and back the TCP window
shut. A timer on the node's own thread does the decode/convert/publish at
`publish_rate_hz`. This is a hard-won fix: with the old single-thread
recv->decode->publish loop, an RELIABLE-QoS rviz/rqt subscription made
`publish()` block for seconds per call once its history queue filled, which
silently throttled the Pi to ~130 KB/s over what looked -- from RTT alone --
like a flaky WiFi link. It wasn't; the socket was just never being drained.
"""

import socket
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

HEADER = struct.Struct(">L")

# Every raw-image consumer in this repo (gdino_detector_node.py,
# sam2_segmenter_node.py) already subscribes with qos_profile_sensor_data
# (BEST_EFFORT). Matching that here is what makes publish() non-blocking --
# a RELIABLE publisher queues undelivered messages for slow subscribers
# instead of dropping them, which is exactly the mechanism that stalled this
# node. depth=1 because only the newest frame is ever useful for a live feed.
IMAGE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class GlobalCamBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_bridge")

        self.declare_parameter("bind_address", "0.0.0.0")
        self.declare_parameter("port", 5000)
        self.declare_parameter("frame_id", "global_cam_optical_frame")
        self.declare_parameter("intrinsics_yaml", "")
        # Above the Pi's measured ~10 fps so the timer never limits us --
        # the bound that matters is however fast the Pi actually sends.
        self.declare_parameter("publish_rate_hz", 15.0)
        # CameraInfo consumers disagree on QoS across this codebase --
        # global_cam_calibrator/localizer/survey all subscribe RELIABLE
        # (depth 10, the create_subscription default), while
        # global_cam_projector uses BEST_EFFORT. A RELIABLE writer serves
        # both (a BEST_EFFORT reader accepts a RELIABLE writer fine), so
        # CameraInfo -- unlike Image -- stays on the default QoS.
        self.declare_parameter("publish_raw", True)
        self.declare_parameter("timestamp_offset_sec", 0.0)
        self.declare_parameter("max_frame_bytes", 8 * 1024 * 1024)
        self.declare_parameter("status_period_sec", 5.0)
        self.declare_parameter("stale_warn_sec", 2.0)

        gp = self.get_parameter
        self.frame_id = str(gp("frame_id").value)
        self.publish_raw = bool(gp("publish_raw").value)
        self.timestamp_offset = Duration(seconds=float(gp("timestamp_offset_sec").value))
        self.max_frame_bytes = int(gp("max_frame_bytes").value)
        self.status_period_sec = float(gp("status_period_sec").value)
        self.stale_warn_sec = float(gp("stale_warn_sec").value)
        publish_rate_hz = float(gp("publish_rate_hz").value)

        self.camera_info = self._load_intrinsics(str(gp("intrinsics_yaml").value))
        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, "/global_cam/image_raw", IMAGE_QOS)
        self.compressed_pub = self.create_publisher(
            CompressedImage, "/global_cam/image_raw/compressed", IMAGE_QOS)
        self.info_pub = self.create_publisher(CameraInfo, "/global_cam/camera_info", 10)

        # -- single-slot handoff between the receive thread and the publish
        # timer. The receive thread only ever writes; the timer only ever
        # reads-and-clears. A lock is enough -- no queue, because a live
        # feed only ever wants the newest frame.
        self._lock = threading.Lock()
        self._latest: Optional[tuple] = None  # (jpeg_bytes, stamp_msg)
        self._latest_seq = 0
        self._last_published_seq = -1
        self._resolution_checked = self.camera_info is None

        # stats, read/written from both threads but only ever incremented --
        # safe enough without a lock for a status line printed a few times a
        # minute.
        self.frames_received = 0
        self.bytes_received = 0
        self.frames_published = 0
        self.bad_decodes = 0
        self.length_rejected = 0
        self._last_recv_wall = 0.0
        self._prev_received = 0
        self._prev_published = 0
        self._prev_bad = 0

        self.bind_addr = (str(gp("bind_address").value), int(gp("port").value))
        self.srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv_sock.bind(self.bind_addr)
        self.srv_sock.listen(1)
        self.srv_sock.settimeout(1.0)  # so the accept loop can check self.running

        self.running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

        self.create_timer(1.0 / publish_rate_hz, self._publish_timer_cb)
        self.create_timer(self.status_period_sec, self._status_cb)

        self.get_logger().info(
            f"listening for TCP JPEG stream on tcp://{self.bind_addr[0]}:{self.bind_addr[1]}")

    def _load_intrinsics(self, path: str) -> Optional[CameraInfo]:
        if not path:
            self.get_logger().warning(
                "no intrinsics_yaml -- CameraInfo not published. Ray-casting will "
                "not work until you run the camera_calibration tool "
                "(ros2 run camera_calibration cameracalibrator ...) and point "
                "this parameter at the resulting ost.yaml.")
            return None
        with open(path) as f:
            d = yaml.safe_load(f)

        def matrix_data(value):
            if isinstance(value, dict):
                return [float(v) for v in value["data"]]
            return [float(v) for v in np.asarray(value).reshape(-1)]

        info = CameraInfo()
        info.width = int(d["image_width"])
        info.height = int(d["image_height"])
        info.k = matrix_data(d["camera_matrix"])
        info.d = matrix_data(d["distortion_coefficients"])
        info.distortion_model = d.get("distortion_model", "plumb_bob")

        if "rectification_matrix" in d:
            info.r = matrix_data(d["rectification_matrix"])
        else:
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        if "projection_matrix" in d:
            info.p = matrix_data(d["projection_matrix"])
        else:
            p = np.zeros((3, 4))
            p[:3, :3] = np.asarray(info.k).reshape(3, 3)
            info.p = [float(v) for v in p.reshape(-1)]

        return info

    def _recv_exact(self, conn: socket.socket, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return None  # peer closed
            buf += chunk
        return bytes(buf)

    def _accept_loop(self) -> None:
        while self.running and rclpy.ok():
            try:
                conn, peer = self.srv_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            conn.settimeout(1.0)
            # A bigger kernel buffer gives the DDS-decoupled design more
            # slack to absorb bursts before the Pi's own send buffer fills.
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            self.get_logger().info(f"Pi connected from {peer}")
            self._client_loop(conn)
            self.get_logger().warning(
                "client disconnected -- waiting for reconnect", throttle_duration_sec=5.0)

    def _client_loop(self, conn: socket.socket) -> None:
        try:
            while self.running and rclpy.ok():
                hdr = self._recv_exact(conn, HEADER.size)
                if hdr is None:
                    return  # disconnected -- back to accept()

                (length,) = HEADER.unpack(hdr)
                if length <= 0 or length > self.max_frame_bytes:
                    # A framing desync (or a Pi sending far larger frames
                    # than expected) would otherwise make the next
                    # _recv_exact allocate towards this many bytes with no
                    # timeout. Safer to drop the connection and resync via a
                    # fresh accept() than to trust an unbounded length.
                    self.length_rejected += 1
                    self.get_logger().error(
                        f"rejecting frame: length={length} outside (0, "
                        f"{self.max_frame_bytes}] -- closing connection to resync",
                        throttle_duration_sec=5.0)
                    return

                payload = self._recv_exact(conn, length)
                if payload is None:
                    return

                stamp = (self.get_clock().now() + self.timestamp_offset).to_msg()
                with self._lock:
                    self._latest = (payload, stamp)
                    self._latest_seq += 1
                self.frames_received += 1
                self.bytes_received += len(payload)
                self._last_recv_wall = time.time()
        finally:
            conn.close()

    def _publish_timer_cb(self) -> None:
        with self._lock:
            latest = self._latest
            seq = self._latest_seq
        if latest is None or seq == self._last_published_seq:
            if self._last_recv_wall and (time.time() - self._last_recv_wall) > self.stale_warn_sec:
                self.get_logger().warning(
                    f"no frame received for {time.time() - self._last_recv_wall:.1f}s "
                    "-- is stream_sender_tcp.py running on the Pi, and does its "
                    "--host point at this machine?",
                    throttle_duration_sec=self.stale_warn_sec)
            return
        self._last_published_seq = seq

        payload, stamp = latest
        # Decode only when something needs the pixels: the raw Image
        # publisher, or the one-time resolution sanity check against
        # intrinsics. The compressed topic forwards `payload` untouched, so
        # with publish_raw:=false and the check already done, a frame can go
        # straight from socket to wire with no decode at all.
        need_decode = self.publish_raw or not self._resolution_checked
        frame = None
        if need_decode:
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.bad_decodes += 1
                return

        if not self._resolution_checked:
            self._resolution_checked = True
            h, w = frame.shape[:2]
            if self.camera_info is not None and (w, h) != (self.camera_info.width, self.camera_info.height):
                # Intrinsics are resolution-specific. If the Pi streams at a
                # size other than what the calibration was shot at, K is
                # silently wrong by a scale factor and every downstream
                # distance comes out proportionally off, with no other
                # symptom -- so say so loudly, once.
                self.get_logger().error(
                    f"Pi is streaming {w}x{h} but intrinsics_yaml was calibrated "
                    f"at {self.camera_info.width}x{self.camera_info.height} -- "
                    "camera_matrix is now wrong by a scale factor. Re-run "
                    "camera_calibration at the Pi's actual resolution.")

        if self.publish_raw and frame is not None:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.frame_id
            self.image_pub.publish(img_msg)

        comp_msg = CompressedImage()
        comp_msg.header.stamp = stamp
        comp_msg.header.frame_id = self.frame_id
        comp_msg.format = "jpeg"
        comp_msg.data = payload
        self.compressed_pub.publish(comp_msg)

        if self.camera_info is not None:
            info = self.camera_info
            info.header.stamp = stamp
            info.header.frame_id = self.frame_id
            self.info_pub.publish(info)

        self.frames_published += 1

    def _status_cb(self) -> None:
        period = self.status_period_sec
        recv_fps = (self.frames_received - self._prev_received) / period
        pub_fps = (self.frames_published - self._prev_published) / period
        new_bad = self.bad_decodes - self._prev_bad
        self._prev_received = self.frames_received
        self._prev_published = self.frames_published
        self._prev_bad = self.bad_decodes

        if self.frames_received == 0:
            return  # the stale-frame warning in the publish timer covers this

        age = time.time() - self._last_recv_wall if self._last_recv_wall else float("inf")
        self.get_logger().info(
            f"recv={recv_fps:.1f}fps pub={pub_fps:.1f}fps "
            f"total_recv={self.frames_received} total_pub={self.frames_published} "
            f"bad_decodes={new_bad} (total {self.bad_decodes}) "
            f"rejected={self.length_rejected} last_frame_age={age:.1f}s")

    def destroy_node(self) -> None:
        self.running = False
        try:
            self.srv_sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
