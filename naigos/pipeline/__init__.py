"""Cloud-scheduled candidate learning: snapshot -> train -> held-out eval -> gated promotion.

Everything under this package is stdlib-only decision logic that can be tested
without Modal, a GPU, a network or a populated cache. The Modal wrapper that
schedules it lives in ``naigos/rl/modal_pipeline.py`` and stays thin.

Scope, unchanged from the rest of the repository: this is a simulation. The
pipeline refreshes allowlisted, non-sensitive simulation inputs and retrains a
purely evasive blue policy against parameterized threats. It never modifies a
deployed policy mid-rollout, never adds a source, and never runs unreviewed
external data into training.
"""
