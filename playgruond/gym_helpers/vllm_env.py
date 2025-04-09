from typing import Optional
import numpy as np
import gymnasium as gym

class vllm_env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple environment for the VLLM model.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Optional[dict] = None):
        super(vllm_env, self).__init__()
        self.config = config if config else {}
        self.action_space = gym.spaces.Box(
            low=0.0, high=float(config['dec_max']), shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
            "ave_term_length": gym.spaces.Box(
                low=1.0, high=float(config['max_model_len']), shape=(1,), dtype=np.float32
            ),
            "num_seq": gym.spaces.Box(
                low=0, high=float(config['num_blocks']), shape=(1,), dtype=np.int32
            ),
            }
        )
        self.state 

    def step(self, action):
        # return the action number
        return action

    def reset(self):
        super().reset()
        return 
