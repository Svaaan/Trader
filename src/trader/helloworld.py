"""Talking to HelloWorldAi: send a dataset, wait, collect the model.

The whole exchange is four calls, and the submitter key is the account. Anyone
holding that key can see these jobs and download the models they produce, so it
is read from the environment and never written to disk by this module.

    TRADER_HELLOWORLD_URL    default https://artificialintelligentduck.duckdns.org
    TRADER_SUBMITTER_KEY     required to submit or collect anything

The key is the same one the HelloWorldAi workspace keeps in the browser. Export
the key file from that page and put the `submitterKey` value here, and this
project's jobs appear in the same workspace -- the UI there and the UI here are
two views of the same thing, which is the point of running a real workload
through it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://artificialintelligentduck.duckdns.org"

# Datasets are slow to send and models slow to build; the rest is small.
UPLOAD_TIMEOUT = 900
BUNDLE_TIMEOUT = 300
DEFAULT_TIMEOUT = 60


class HelloWorldError(Exception):
    """Any failure talking to the coordinator, with the reason it gave."""


@dataclasses.dataclass
class Job:
    """One training job, as the coordinator sees it."""

    task_id: str
    status: str
    raw: dict

    @property
    def finished(self) -> bool:
        return self.status in ("completed", "failed", "rejected", "cancelled")

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and bool(self.raw.get("weights_id"))

    @property
    def verification(self) -> dict:
        """What the coordinator measured, not what the node reported.

        These are separate for a reason worth preserving here: `metrics` is the
        node describing its own work, `verification` is the coordinator
        rebuilding the model from the returned weights and scoring it on rows
        the node never saw. Only the second is evidence.
        """
        return self.raw.get("verification") or {}

    @property
    def model_name(self) -> Optional[str]:
        return (self.raw.get("task_data") or {}).get("model_name")


class Client:
    def __init__(self, base_url: str | None = None, submitter_key: str | None = None):
        self.base_url = (base_url or os.environ.get("TRADER_HELLOWORLD_URL")
                         or DEFAULT_URL).rstrip("/")
        self.submitter_key = submitter_key or os.environ.get("TRADER_SUBMITTER_KEY")

    # --- plumbing ----------------------------------------------------------

    def _headers(self, extra: dict | None = None) -> dict:
        if not self.submitter_key:
            raise HelloWorldError(
                "No submitter key. Set TRADER_SUBMITTER_KEY to the `submitterKey` "
                "from your HelloWorldAi key file -- without it the coordinator "
                "cannot tell whose jobs these are.")
        headers = {"X-Submitter-Key": self.submitter_key}
        headers.update(extra or {})
        return headers

    def _call(self, method: str, path: str, *, timeout: float = DEFAULT_TIMEOUT,
              content: bytes | None = None, json_body: Any = None,
              headers: dict | None = None, raw_response: bool = False):
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.request(
                    method, url, content=content, json=json_body,
                    headers=self._headers(headers))
        except httpx.RequestError as exc:
            raise HelloWorldError(f"Could not reach {url}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:400]
            try:
                body = response.json()
                detail = body.get("detail") or body.get("error") or detail
            except Exception:                              # noqa: BLE001
                pass
            raise HelloWorldError(
                f"{method} {path} returned {response.status_code}: {detail}")

        if raw_response:
            return response.content
        try:
            return response.json()
        except Exception as exc:                           # noqa: BLE001
            raise HelloWorldError(
                f"{method} {path} did not return JSON: {response.text[:200]}") from exc

    # --- the four calls ----------------------------------------------------

    def upload_dataset(self, blob: bytes) -> str:
        """Store the .npz and get back an id to submit against."""
        body = self._call("POST", "/artifacts", content=blob,
                          timeout=UPLOAD_TIMEOUT,
                          headers={"Content-Type": "application/octet-stream"})
        artifact_id = body.get("artifact_id")
        if not artifact_id:
            raise HelloWorldError(f"Upload returned no artifact id: {body}")
        logger.info("Uploaded %d bytes as %s", len(blob), artifact_id)
        return artifact_id

    def nodes(self) -> list[dict]:
        """Every node the coordinator knows about, from the database."""
        body = self._call("GET", "/nodes")
        return body if isinstance(body, list) else []

    def pick_node(self, *, max_silence_seconds: int = 300) -> Optional[str]:
        """The node most recently heard from, or None to let the coordinator choose.

        The coordinator's own placement trusts `isConnected`, and that flag is
        not reconciled against the heartbeat: a machine that stopped reporting
        ten minutes ago still reads as connected and still gets work. Measured
        on the live network -- a job was assigned to a node whose last heartbeat
        was ten minutes older than the other node's, and sat pending
        indefinitely, because the stale-task reaper only rescues jobs that
        reached `running`.

        So this picks by the one field that cannot go stale without being
        noticed: how long ago the node last spoke.
        """
        import datetime as dt

        best, best_age = None, None
        now = dt.datetime.now(dt.timezone.utc)

        for node in self.nodes():
            if not node.get("isAvailable"):
                continue
            stamp = node.get("last_heartbeat")
            if not stamp:
                continue
            try:
                seen = dt.datetime.fromisoformat(str(stamp))
                if seen.tzinfo is None:
                    seen = seen.replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue

            age = (now - seen).total_seconds()
            if age > max_silence_seconds:
                logger.info("Skipping %s: silent for %.0fs",
                            node.get("node_id"), age)
                continue
            if best_age is None or age < best_age:
                best, best_age = node.get("node_id"), age

        if best:
            logger.info("Chose %s (heard from %.0fs ago)", best, best_age)
        return best

    def cancel(self, task_id: str) -> None:
        self._call("POST", f"/cancel-task/{task_id}")

    def submit(self, *, dataset_id: str, model_name: str,
               steps: int = 4000, batch_size: int = 64,
               learning_rate: float = 0.01,
               hidden_dim: int = 64, depth: int = 2,
               node_id: str | None = None,
               time_ordered: bool = True) -> str:
        """Queue a job, on a named node when one has been chosen.

        Naming a node is not the default in HelloWorldAi and for good reason --
        it queues behind whatever that machine is doing. It is the default here
        because auto-placement picks on a stale flag; see pick_node.
        """
        path = f"/submit-task/{node_id}" if node_id else "/submit-task"
        # Everything this project sends is a price series, so the coordinator's
        # verification should hold back the newest rows rather than a random
        # slice. A random slice of a market grades the model on days it has both
        # neighbours of, which is a much easier question than the one being
        # asked -- measured at 54.2% against 51.7% for the same weights.
        body = self._call("POST", path, json_body={
            "time_ordered": time_ordered,
            "model_name": model_name,
            "architecture": "mlp",          # rows of numbers to a class
            "dataset_id": dataset_id,
            "hyperparameters": {
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "hidden_dim": hidden_dim,
            "depth": depth,
        })
        task_id = body.get("task_id") or (body.get("task") or {}).get("task_id")
        if not task_id:
            raise HelloWorldError(f"submit-task returned no task id: {body}")
        logger.info("Submitted %s as %s", model_name, task_id)
        return task_id

    def jobs(self, limit: int = 25) -> list[Job]:
        """Every job sent with this key, newest first."""
        body = self._call("GET", f"/my-tasks?limit={limit}")
        if not isinstance(body, list):
            raise HelloWorldError(f"/my-tasks did not return a list: {body}")
        return [Job(task_id=j.get("task_id"), status=j.get("status", "unknown"),
                    raw=j) for j in body]

    def job(self, task_id: str) -> Optional[Job]:
        for candidate in self.jobs(limit=100):
            if candidate.task_id == task_id:
                return candidate
        return None

    def download_bundle(self, task_id: str) -> bytes:
        """The finished model: a zip of weights, config and a README."""
        blob = self._call("GET", f"/my-tasks/{task_id}/bundle",
                          timeout=BUNDLE_TIMEOUT, raw_response=True)
        if len(blob) < 200:
            raise HelloWorldError(
                f"The bundle for {task_id} is {len(blob)} bytes, which is not a model")
        logger.info("Downloaded %d bytes for %s", len(blob), task_id)
        return blob

    # --- convenience -------------------------------------------------------

    def reachable(self) -> tuple[bool, str]:
        """For the UI, which should say what is wrong rather than sit blank."""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.base_url}/get-connected-nodes-count")
            if response.status_code != 200:
                return False, f"{self.base_url} answered {response.status_code}"
            count = (response.json() or {}).get("connected_nodes_count")
            return True, f"{count} node(s) online"
        except Exception as exc:                           # noqa: BLE001
            return False, f"could not reach {self.base_url}: {exc}"
