from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
        )
