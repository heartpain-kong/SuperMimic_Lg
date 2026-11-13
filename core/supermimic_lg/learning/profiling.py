"""Stage timings without synchronizing CUDA between learner stages."""
from contextlib import contextmanager
import time
import torch


class LearnerTimings:
    names = ('actor_rollout', 'actor_backward', 'critic_targets', 'critic_fit', 'policy_ema')

    def __init__(self, device):
        self.device = torch.device(device)
        self.pending = []

    def reset(self):
        self.pending.clear()

    @contextmanager
    def measure(self, name):
        if self.device.type == 'cuda':
            stream = torch.cuda.current_stream(self.device)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record(stream)
        else:
            start = time.perf_counter()
        try:
            with torch.profiler.record_function('mbrl/'+name):
                yield
        finally:
            if self.device.type == 'cuda':
                end.record(stream)
            else:
                end = time.perf_counter()
            self.pending.append((name, start, end))

    def metrics(self, synchronize=True):
        # The runner already synchronizes once at the iteration boundary. Tests
        # and standalone benchmarks can request the synchronization explicitly.
        if self.device.type == 'cuda' and self.pending and synchronize:
            torch.cuda.synchronize(self.device)
        result = {'time/'+name+'_s': 0. for name in self.names}
        for name, start, end in self.pending:
            elapsed = start.elapsed_time(end)/1000. if self.device.type == 'cuda' else end-start
            result['time/'+name+'_s'] += elapsed
        self.reset()
        return result
