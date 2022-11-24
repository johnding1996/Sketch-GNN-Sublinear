import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.timings = [[] for _ in range(runs)]
        self.memories = [[] for _ in range(runs)]

    def add_result(self, run, result, timing, memory):
        assert len(result) == 3
        assert len(timing) == 3
        assert len(memory) == 2
        assert 0 <= run < len(self.results)
        self.results[run].append(result)
        self.timings[run].append(timing)
        self.memories[run].append(memory)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            timing = torch.tensor(self.timings[run])
            avgtime = timing[1:, :].mean(dim=0)
            memory = torch.tensor(self.memories[run])
            peakmem = memory.max(dim=0)[0]

            print('#' * 60)
            print(f'Run {run + 1:02d}:')
            print(f'           Highest Train: {result[:, 0].max():05.2f}%')
            print(f'           Highest Valid: {result[:, 1].max():05.2f}%')
            print(f'             Final Train: {result[argmax, 0]:05.2f}%')
            print(f'              Final Test: {result[argmax, 2]:05.2f}%')
            print(f' Average Load Epoch Time: {avgtime[0]:06.4f}s')
            print(f'Average Train Epoch Time: {avgtime[1]:06.4f}s')
            print(f' Average Test Epoch Time: {avgtime[2]:06.4f}s')
            print(f'  Peak Test Memory Usage: {peakmem[0] / 1048576.0:07.2f}MB')
            print(f' Peak Train Memory Usage: {peakmem[1] / 1048576.0:07.2f}MB')
            print('#' * 60)
        else:
            results = 100 * torch.tensor(self.results)
            best_results = []
            for r in results:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))
            best_result = torch.tensor(best_results)
            timings = torch.tensor(self.timings)
            avgtime = timings[:, 1:, :].mean(dim=(0, 1))
            memories = torch.tensor(self.memories)
            peakmem = memories.max(dim=1)[0].max(dim=0)[0]
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'           Highest Train: {r.mean():.2f}% ± {r.std():05.2f}%')
            r = best_result[:, 1]
            print(f'           Highest Valid: {r.mean():.2f}% ± {r.std():05.2f}%')
            r = best_result[:, 2]
            print(f'             Final Train: {r.mean():.2f}% ± {r.std():05.2f}%')
            r = best_result[:, 3]
            print(f'              Final Test: {r.mean():.2f}% ± {r.std():05.2f}%')
            print(f' Average Load Epoch Time: {avgtime[0]:06.4f}s')
            print(f'Average Train Epoch Time: {avgtime[1]:06.4f}s')
            print(f' Average Test Epoch Time: {avgtime[2]:06.4f}s')
            print(f'  Peak Test Memory Usage: {peakmem[0] / 1048576.0:07.2f}MB')
            print(f' Peak Train Memory Usage: {peakmem[1] / 1048576.0:07.2f}MB')
            print('#' * 60)
