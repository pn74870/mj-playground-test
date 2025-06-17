# mj-playground-test
testing new playground for RL

## Custom Environments

`learning/baby_freeze_env.py` contains an example of extending `mujoco_playground` to create a custom `HumanoidBabyFreeze` environment. This environment encourages a humanoid to maintain a static freeze pose. After installing the playground dependencies you can register and load it via:

```python
from mujoco_playground import registry
import learning.baby_freeze_env  # Registers the env
env = registry.load("HumanoidBabyFreeze")
```

Training can then follow the procedure in `learning/notebooks/dm_control_suite.ipynb`.
