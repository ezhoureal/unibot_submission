"""Serve LingBotPolicy over websocket for the UniBot evaluator to connect to.

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint python run_server.py [port]
"""

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lingbot_policy import LingBotPolicy
from policy.web_policy import PolicyService


def main(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Build the policy and serve it on host:port."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [server] %(message)s",
    )
    policy = LingBotPolicy()
    safe_meta = {k: v for k, v in policy.metadata.items() if k != "token"}
    print(f"Serving LingBot-VLA 2.0 LoRA policy on ws://{host}:{port}")
    print("  control_space     =", safe_meta["control_space"])
    print("  data_keys         =", safe_meta["data_keys"])
    print("  obs_chunk_size    =", safe_meta["obs_chunk_size"])
    print("  action_chunk_size =", safe_meta["action_chunk_size"])
    PolicyService(policy, host=host, port=port).run_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    main(port=port)
