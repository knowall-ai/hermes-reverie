"""Reverie is a generic plugin: nothing an installed agent reads may name one deployment.

Guards CONTRIBUTING.md rule 1 after skills/dreaming/SKILL.md shipped "ask Sallie in Teams"
and "graph.mjs user lookup" (hermes-reverie #6).
"""
import pathlib
import re

import pytest
from conftest import load_plugin

reverie = load_plugin()
ROOT = pathlib.Path(reverie.__file__).resolve().parent
DEPLOYMENT_NAMES = re.compile(
    r"\b(sallie|poppie|knowall|graph\.mjs|azure devops|openclaw|ben weeks)\b", re.IGNORECASE)


def _hits(text: str):
    return sorted({m.group(0).lower() for m in DEPLOYMENT_NAMES.finditer(text)})


def test_tool_description_is_deployment_agnostic():
    assert _hits(reverie.REVERIE_TOOL["description"]) == []
    for prop in reverie.REVERIE_TOOL["parameters"]["properties"].values():
        assert _hits(prop.get("description", "")) == []


def test_session_prompt_block_is_deployment_agnostic():
    class _Stats:
        def stats(self):
            return {"nodes": 3}
    provider = reverie.ReverieMemoryProvider.__new__(reverie.ReverieMemoryProvider)
    provider._graph = _Stats()
    assert _hits(provider.system_prompt_block()) == []


@pytest.mark.parametrize("skill", sorted((ROOT / "skills").glob("*/SKILL.md")),
                         ids=lambda p: p.parent.name)
def test_skills_are_deployment_agnostic(skill):
    assert _hits(skill.read_text()) == [], f"{skill.relative_to(ROOT)} names a deployment"


def test_guard_catches_a_deployment_name():
    assert _hits("ask Sallie in Teams, or use graph.mjs") == ["graph.mjs", "sallie"]
