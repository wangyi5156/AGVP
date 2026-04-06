r"""BaselineRegistry is extended from habitat.Registry to provide
registration for trainer and environments, while keeping Registry
in habitat core intact.

Import the baseline registry object using

``from av_nav.common.baseline_registry import baseline_registry``

Various decorators for registry different kind of classes with unique keys

- Register a environment: ``@registry.register_env``
- Register a trainer: ``@registry.register_trainer``
- Register a policy: ``@registry.register_policy``
"""

from typing import Optional

from habitat.core.registry import Registry
from av_nav.rl.ppo.policy import PointNavBaselinePolicy  # **需要导入**

class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'."""
        from av_nav.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register an environment to registry with key 'name'."""
        return cls._register_impl("env", to_register, name)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)

    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a policy network to registry with key 'name'."""
        from av_nav.rl.ppo.policy import Policy  # Or use a base Net class

        return cls._register_impl(
            "policy", to_register, name, assert_type=None
        )

    @classmethod
    def get_policy(cls, name):
        return cls._get_impl("policy", name)


baseline_registry = BaselineRegistry()

# 这里确保 PointNavBaselinePolicy 已导入，否则报错
baseline_registry.register_policy(PointNavBaselinePolicy, name="PointNavBaselinePolicy")

