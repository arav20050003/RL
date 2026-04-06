# risk/__init__.py
# STATUS: Package init — exports risk signal functions
# DEPENDS ON: llm_risk_signal.py
# TEST: python -c "from risk import get_risk_score; print(get_risk_score(1))"

from risk.llm_risk_signal import get_risk_score, get_risk_score_batch

__all__ = ["get_risk_score", "get_risk_score_batch"]
