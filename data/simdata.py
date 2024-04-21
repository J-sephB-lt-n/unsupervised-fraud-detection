"""

"""

import itertools
import random
from typing import Final

N_PEOPLE: Final[int] = 10


class Person:
    def __init__(self, pid: str) -> None:
        self.pid = pid
        self.pay_range = (
            round(random.uniform(0, 500), 2),
            round(random.uniform(500, 1_000), 2),
        )
        times_cycler = itertools.cycle(range(1, 25))
        [next(times_cycler) for _ in range(random.randint(0, 23))]
        self.time_range = [next(times_cycler) for _ in range(random.randint(4, 12))]
        self.payees: list[tuple] = []

    def transact(self) -> None:



all_people: list[Person] = [
    Person(pid=f"p{idx}") for idx in range(1, N_PEOPLE + 1)
]

for person in all_people:
    print(person.pid, person.pay_range, person.time_range) 
