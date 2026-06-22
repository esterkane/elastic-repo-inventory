"""Optional grounded-generation path, gated on faithfulness.

Generation is to faithfulness what search is to relevance: an opt-in
(``GENERATION_ENABLED``, default false) answer path that retrieves evidence,
drafts an answer grounded ONLY in the retrieved chunks via the swappable
provider, attaches a retained ``source_url`` to every claim, and drops any
claim that does not trace — reusing the agents' ``WriterAgent`` /
``VerifierAgent`` so generation enforces exactly the provenance the eval
measures.
"""

from backend.app.generation.answer import answer, generation_enabled

__all__ = ["answer", "generation_enabled"]
