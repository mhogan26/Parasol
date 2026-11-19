import csv
from dataclasses import dataclass
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@dataclass
class DataReporterFrame:
    reward: float
    background_packets: int
    budget: int
    action: int


class DataReporter:
    def __init__(self):
        self.frames: list(DataReporterFrame) = []
        self.episode = 0

    def add_frame(self, reward, background, budget, action):
        self.frames.append(
            DataReporterFrame(
                reward=reward,
                background_packets=int(background),
                budget=int(budget),
                action=int(action),
            )
        )

    def report(self, outfile=None):
        # TODO figure out what other metrics to display
        # Broadly, my methodology is to display metrics and derived stats
        # to stderr/the terminal, and send more detailed data to a file for
        # later analysis.
        errs = [x.avg_err for x in self.frames]
        actions = [x.action for x in self.frames]
        avg_err = sum(errs) / len(errs)
        eprint("Overall average error: {}".format(avg_err))
        eprint("Actions that led to this error:\n{}".format(actions))
        if outfile:
            eprint("Sending data to {}...".format(outfile), end='')
            with open(outfile, 'w', newline='') as csvfile:
                fieldnames = ['reward', 'background_packets', 'budget', 'action']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(map(vars, self.frames))
            eprint("done!")

    def start_or_advance_step(self, episodes):
        if self.episode < 1:
            eprint()
        self.episode += 1
        eprint(
            f"\rTraining: episode {self.episode}/{episodes}",
            end='', flush=True
        )
        if self.episode == episodes:
            eprint()
