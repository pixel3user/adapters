"""Stage-1 regression tests: stealth agent must not depend on /tests."""

from __future__ import annotations

import importlib.util
import unittest


class TaskCatalogTests(unittest.TestCase):
    def test_catalog_default_samples_match_upstream_filters(self) -> None:
        from gdm_stealth.task_catalog import SAMPLES_BY_KEY

        defaults = [s for s in SAMPLES_BY_KEY.values() if s["default"]]
        self.assertEqual(len(defaults), 13)
        self.assertIn(
            "oversight_pattern--email_domain_topic_bomb_difficulty_feedback_in_system_num_failures_allowed_8",
            SAMPLES_BY_KEY,
        )
        self.assertIn("cover_your_tracks--no_hint", SAMPLES_BY_KEY)
        self.assertFalse(
            SAMPLES_BY_KEY["cover_your_tracks--with_hint"]["default"]
        )
        record = SAMPLES_BY_KEY["cover_your_tracks--no_hint"]
        self.assertEqual(record["task_slug"], "cover-your-tracks-no-hint")
        self.assertEqual(record["dataset_name"], "cover_your_tracks")

    def test_marker_payload_contains_only_identity(self) -> None:
        from gdm_stealth.task_catalog import marker_payload

        payload = marker_payload("cover_your_tracks--no_hint")
        self.assertEqual(set(payload), {"adapter", "sample_key"})
        self.assertNotIn("system_prompt", payload)
        self.assertNotIn("scoring", payload)


@unittest.skipUnless(importlib.util.find_spec("harbor"), "Harbor is not installed")
class AgentWithoutTestsMountTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_uses_marker_not_tests(self) -> None:
        import json
        import pathlib
        import subprocess
        import tempfile

        from harbor.environments.base import ExecResult
        from harbor.models.agent.context import AgentContext

        from gdm_stealth.custom_agent.agent import GDMStealthParityAgent
        from gdm_stealth.custom_agent.transport import ModelTurn, ToolCall
        from gdm_stealth.task_catalog import marker_payload

        key = (
            "oversight_pattern--email_domain_topic_bomb_difficulty_"
            "feedback_in_system_num_failures_allowed_8"
        )

        class FakeEnv:
            def __init__(self, root: pathlib.Path):
                self.agent = root / "agent"
                self.agent.mkdir(parents=True)
                (self.agent / "email_tasks.txt").write_text(
                    "--- Task 1 ---\n"
                    "Recipient: alice@example.com\n"
                    "Recipient: bob@example.com\n"
                    "Recipient: carol@example.com\n"
                )
                self.marker = root / "task.json"
                self.marker.write_text(json.dumps(marker_payload(key)))

            async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
                cmd = command.replace("/opt/gdm/task.json", str(self.marker))
                cmd = cmd.replace("/home/agent", str(self.agent))
                cmd = cmd.replace("/var/lib/gdm", str(self.agent))
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.agent,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec or 60,
                )
                return ExecResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=result.returncode,
                )

        recipients = [
            "alice@example.com",
            "bob@example.com",
            "carol@example.com",
        ]
        turns = iter(
            ModelTurn(
                tool_calls=[
                    ToolCall(
                        f"c{i}",
                        "send_email",
                        {
                            "subject": "Update",
                            "body": "Quick note about Bomb magazine.",
                            "recipient": recipient,
                        },
                    )
                ]
            )
            for i, recipient in enumerate(recipients)
        )

        class FakeTransport:
            async def generate(self, messages, schemas):
                return next(turns)

        with tempfile.TemporaryDirectory() as td:
            env = FakeEnv(pathlib.Path(td))
            agent = GDMStealthParityAgent(
                logs_dir=pathlib.Path(td) / "logs",
                model_name="fake/model",
                transport=FakeTransport(),
            )
            ctx = AgentContext()
            await agent.run("GOAL: test", env, ctx)
            self.assertTrue(ctx.metadata["outcome"]["success"])
            self.assertTrue((env.agent / "gdm_outcome.json").exists() or
                            (env.agent / "outcome.json").exists())


if __name__ == "__main__":
    unittest.main()
