import time


def format_clock(period):
    hour, minute, second = period // 3600, (period % 3600) // 60, period % 60
    return int(hour), int(minute), int(second)


class Timer:
    def __init__(self, total: int):
        self.total = total
        self.ticktock = 0
        self.last = None
        self.avg_time = 0

    def step(self):
        if self.last is not None:
            period = time.time() - self.last
            self.avg_time = (self.avg_time * (self.ticktock - 1) + period) / self.ticktock
            h1, m1, s1 = format_clock(self.avg_time * (self.ticktock + 1))
            h2, m2, s2 = format_clock(self.avg_time * (self.total - self.ticktock))
            print(
                f"STEP {self.ticktock}/{self.total} | USED: %02d:%02d:%02d | VAG %.2f s/it | "
                f"ETA: %02d:%02d:%02d" % (h1, m1, s1, self.avg_time, h2, m2, s2)
            )
        self.last = time.time()
        self.ticktock += 1
        if self.ticktock == self.total:
            self.reset()

    def reset(self):
        self.ticktock = 0
        self.last = None
        self.avg_time = 0


if __name__ == '__main__':
    timer = Timer(100)
    for _ in range(100):
        time.sleep(1)
        timer.step()
