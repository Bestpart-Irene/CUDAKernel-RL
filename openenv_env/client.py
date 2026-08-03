"""OpenEnv client for KernelForge — used by TRL and external consumers."""
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from openenv_env.models import KernelForgeAction, KernelForgeObservation


class KernelForgeClient(EnvClient[KernelForgeAction, KernelForgeObservation, State]):
    """HTTP/WebSocket client for the KernelForge environment.

    Implements the three EnvClient hooks (matching the openenv-core 0.2.1
    template client): action -> JSON payload, server payload -> StepResult,
    and state payload -> State.
    """

    def _step_payload(self, action: KernelForgeAction) -> Dict[str, Any]:
        """Convert a KernelForgeAction to the JSON data expected by the env server."""
        return {"cuda_code": action.cuda_code}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[KernelForgeObservation]:
        """Parse a server response into StepResult[KernelForgeObservation].

        The server serializes observations as
        ``{"observation": {...fields...}, "reward": ..., "done": ...}``
        (see openenv.core.env_server.serialization.serialize_observation).
        Field names mirror KernelForgeEnv's observation construction.
        """
        obs_data = payload.get("observation", {})
        observation = KernelForgeObservation(
            text=obs_data.get("text", ""),
            baseline_original_ms=obs_data.get("baseline_original_ms"),
            baseline_doublegraph_ms=obs_data.get("baseline_doublegraph_ms"),
            hardware=obs_data.get("hardware", {}),
            turn=obs_data.get("turn", 0),
            best_reward=obs_data.get("best_reward", -1.0),
            info=obs_data.get("info", {}),
            graph_properties=obs_data.get("graph_properties"),
            topology_type=obs_data.get("topology_type"),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse a state-endpoint response into a State object.

        State allows extra fields, so KernelForgeEnv extras (history,
        best_reward) are passed through.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            **{
                k: v
                for k, v in payload.items()
                if k not in ("episode_id", "step_count")
            },
        )
