# envs/__init__.py
# STATUS: Package init — exports environment classes
# DEPENDS ON: scimai_env.py, disruption_env.py
# TEST: python -c "from envs import ScimaiEnv"

from envs.scimai_env import ScimaiEnv

# DisruptionEnv imported separately to avoid circular dependency during Phase 1
# from envs.disruption_env import DisruptionEnv

__all__ = ["ScimaiEnv"]
