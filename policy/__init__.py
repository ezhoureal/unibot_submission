"""WebSocket transport for serving / consuming a policy.

The transport lives in `web_policy`, which carries both ends of the wire:

  - `web_policy.PolicyService` — wrap a local policy and serve it on a
    websocket. Single-policy deployment only.
  - `web_policy.RemotePolicy`  — connect to such a server and drive it as
    if it were a local policy.

These are generic transport. The observation / action contract is
described in the README and enforced at the policy boundary by
`example/example_env.py` — not in this package.
"""
