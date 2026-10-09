# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to communicate with a Docker based policy server."""

import argparse
import dora
import json
import os
import pyarrow as pa
import shutil
import socket
import subprocess
import tempfile


def _main_dora(io, host_shared_dir, container_shared_dir):
    n_keep_data = 5  # TODO: Customizable?
    data_files = []

    node = dora.Node()
    previous_observation_id = None
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        def prepare_request():
            nonlocal previous_observation_id
            observation = event["value"]
            observation_id = observation.field("id")[0].as_py()
            # The observation id increments per observation and drops back to a
            # lower value when a new episode starts, so a decrease (or the very
            # first observation) signals a reset to the policy server.
            reset = (
                previous_observation_id is None
                or observation_id < previous_observation_id
            )
            previous_observation_id = observation_id
            host_data_file = tempfile.NamedTemporaryFile(
                suffix=".arrow", dir=host_shared_dir, delete_on_close=False
            )
            record_batch = pa.RecordBatch.from_struct_array(observation)
            with pa.output_stream(host_data_file) as output:
                with pa.ipc.new_file(output, record_batch.schema) as writer:
                    writer.write(record_batch)
            data_files.append(host_data_file)
            if len(data_files) > n_keep_data:
                data_files.pop(0)
            container_data_path = os.path.join(
                container_shared_dir, os.path.basename(host_data_file.name)
            )
            return {
                "name": "inference",
                "data_path": container_data_path,
                "reset": reset,
                "metadata": event["metadata"],
            }

        # dora-rs node -> Policy server: Inference request
        #   {"name": "inference", "data_path": "/data/path.arrow",
        #    "reset": true/false, ...}
        #
        # "/data/path.arrow" has a record batch:
        #   {
        #     # element len: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     # "arm_right" + "arm_left"
        #     "position": pa.list_(pa.float32()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_right": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_left": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_head": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_ceiling": pa.list_(pa.uint8()),
        #     }
        #   }
        request = prepare_request()
        io.write(json.dumps(request) + "\n")
        io.flush()

        # Policy server -> dora-rs node: Inferred actions
        #   {
        #     "interval": interval_in_ns,
        #     "positions": [
        #        [...],
        #        ...
        #     ]
        #   }
        #
        #   Arm position: Motor positions
        #     Bimanual: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     Unimanual: 8 (7 joints + 1 gripper)
        response = io.readline()
        if not response:
            break
        actions = json.loads(response)
        if actions["positions"]:
            # "reset" signals that these actions are the first of a new episode,
            # so downstream nodes can drop any state carried over from the
            # previous episode.
            metadata = {"interval": actions["interval"], "reset": request["reset"]}
            if "cutoff_hz" in actions:
                metadata["cutoff_hz"] = actions["cutoff_hz"]
            # The first action is the target for the observation's time, not for
            # now; pass that time on so the executor can skip what is already past.
            if "timestamp" in request["metadata"]:
                metadata["observation_timestamp"] = request["metadata"]["timestamp"]
            node.send_output(
                "actions",
                pa.array(actions["positions"], type=pa.list_(pa.float32())),
                metadata,
            )


def main():
    """Communicate with a Docker based policy server."""
    parser = argparse.ArgumentParser(
        description="Communicate with a Docker based policy server"
    )
    parser.add_argument(
        "--image",
        default=os.getenv("IMAGE"),
        help="The Docker image name",
        type=str,
    )
    parser.add_argument(
        "--volume",
        action="append",
        default=[],
        help="The additional volumes for the Docker container",
    )
    parser.add_argument(
        "--pull-policy",
        default=os.getenv("PULL_POLICY", "missing"),
        choices=["missing", "always", "never"],
        help="The Docker image pull policy",
    )
    args = parser.parse_args()

    docker = shutil.which("docker")
    with tempfile.TemporaryDirectory(
        prefix="dora-openarm-docker-policy-server", dir="/dev/shm"
    ) as host_shared_dir:
        container_shared_dir = "/openeval"
        socket_base_path = "connection.sock"
        host_socket_path = os.path.join(host_shared_dir, socket_base_path)
        container_socket_path = os.path.join(container_shared_dir, socket_base_path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(host_socket_path)
            sock.listen()
            command_line = [
                docker,
                "run",
                "--gpus=all",
                "--interactive",
                "--network=none",  # Disable network for security
                f"--pull={args.pull_policy}",
                "--rm",
                f"--volume={host_shared_dir}:{container_shared_dir}:ro",
                "--volume=cache:/cache",
                *[f"--volume={v}" for v in args.volume],
                args.image,
                container_socket_path,
            ]
            try:
                inferencer = subprocess.Popen(command_line, cwd="/")
                with sock.accept()[0] as connection:
                    with connection.makefile("rw") as io:
                        _main_dora(io, host_shared_dir, container_shared_dir)
            finally:
                if inferencer.poll() is None:
                    inferencer.terminate()
                if inferencer.poll() is None:
                    inferencer.kill()
                    inferencer.wait()


if __name__ == "__main__":
    main()
