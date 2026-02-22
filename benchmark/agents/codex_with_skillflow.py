"""
Custom Codex agent that uses SkillFlow for dynamic skill discovery.
"""

from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from benchmark.agents.base import BaseCodexAgent


class CodexWithSkillFlow(BaseCodexAgent):
    """
    Codex agent with SkillFlow integration for dynamic skill discovery.

    Unlike CodexAgent which injects static skills, this agent can query
    a SkillFlow peer during execution to discover and retrieve relevant skills.
    """

    DEFAULT_PEER_URL = "http://172.17.0.1:8765"

    def __init__(
        self,
        *args: Any,
        skillflow_peer_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize CodexWithSkillFlow agent.

        Args:
            skillflow_peer_url: URL of the SkillFlow peer server.
                Defaults to http://172.17.0.1:8765 (Docker gateway).
        """
        super().__init__(*args, **kwargs)
        self._skillflow_peer_url = skillflow_peer_url or self.DEFAULT_PEER_URL

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent with SkillFlow capabilities."""
        await super().setup(environment)
        await self._setup_skillflow(environment)

    async def _setup_skillflow(self, environment: BaseEnvironment) -> None:
        """Install skillflow-client in the container."""
        self.logger.debug("Setting up SkillFlow integration")

        # Upload standalone skillflow-client script to container
        client_script = Path(__file__).parent / "scripts" / "skillflow-client"
        client_content = client_script.read_text()

        # Upload to container and make executable
        upload_cmd = (
            f"cat > /usr/local/bin/skillflow-client << 'SCRIPT_EOF'\n"
            f"{client_content}\nSCRIPT_EOF"
        )
        await environment.exec(command=upload_cmd)
        await environment.exec(command="chmod +x /usr/local/bin/skillflow-client")

        self.logger.debug("SkillFlow setup complete")
