"""
A script for simulating basic transaction data

Script usage:
    $ poetry run python -m data.simdata
"""

import collections
import csv
import itertools
import random
from typing import Final

N_PEOPLE: Final[int] = 5_000
N_FRAUDSTERS: Final[int] = 50
N_PAYEES: Final[tuple[int, int]] = (1, 10)
N_NON_FRAUD_TRANSACTIONS: Final[int] = 100_000
N_FRAUD_TRANSACTIONS: Final[int] = 500
SIM_DATA_OUTPUT_PATH: Final[str] = "data/input/simdata.csv"

FRAUDSTER_IND: Final[set[int]] = set(random.sample(range(N_PEOPLE), k=N_FRAUDSTERS))

Transaction = collections.namedtuple(
    "Transaction", ["src", "dst", "amt", "time", "day_of_week", "is_fraud"]
)


class Person:
    def __init__(self, pid: str) -> None:
        self.pid = pid
        self.n_payees = random.randint(*N_PAYEES)
        self.pay_range: tuple[float, float] = sorted(
            round(random.uniform(1, 1000), 2) for _ in range(2)
        )
        self.receive_range: tuple[float, float] = sorted(
            round(random.uniform(1, 1000), 2) for _ in range(2)
        )
        times_cycler = itertools.cycle(range(1, 25))
        [next(times_cycler) for _ in range(random.randint(0, 23))]
        self.time_range: tuple[int, ...] = tuple(
            [next(times_cycler) for _ in range(random.randint(4, 12))]
        )
        self.pay_day_of_week: tuple[int, ...] = tuple(
            sorted(random.sample(range(1, 8), k=random.randint(1, 4)))
        )
        self.payees: list[Person] = []
        self.is_fraudster: bool = False

    def transact(self) -> Transaction:
        payee = random.choice(self.payees)
        return Transaction(
            src=self.pid,
            dst=payee.pid,
            amt=round(
                random.uniform(
                    max(self.pay_range[0], payee.receive_range[0]),
                    min(self.pay_range[1], payee.receive_range[1]),
                ),
                2,
            ),
            time=random.choice(self.time_range),
            day_of_week=random.choice(self.pay_day_of_week),
            is_fraud=False,
        )

    def __repr__(self) -> str:
        return f""" -- {self.pid} -- 
PAY RANGE: {self.pay_range[0]} - {self.pay_range[1]}
RECEIVE RANGE: {self.receive_range[0]} - {self.receive_range[1]}
PAY TIMES: {self.time_range}
PAY DAY OF WEEK: {self.pay_day_of_week}
IS FRAUDSTER: {self.is_fraudster}
PAYEES: {[p.pid for p in self.payees]}
    """


all_people: list[Person] = [Person(pid=f"p{idx}") for idx in range(1, N_PEOPLE + 1)]

for person_idx, person in enumerate(all_people):
    if person_idx in FRAUDSTER_IND:
        person.is_fraudster = True
    while len(person.payees) < person.n_payees:
        candidate = random.choice(all_people)
        if (
            candidate != person
            and candidate not in person.payees
            and (
                min(person.pay_range[1], candidate.receive_range[1])
                - max(person.pay_range[0], candidate.receive_range[0])
            )
            > 0
        ):
            person.payees.append(candidate)

with open(SIM_DATA_OUTPUT_PATH, mode="w", encoding="utf-8") as file:
    csv_writer = csv.DictWriter(
        file,
        fieldnames=[
            "tid",
            "src",
            "dst",
            "amt",
            "time",
            "day_of_week",
            "is_fraud",
        ],
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )
    csv_writer.writeheader()
    for tid in range(1, N_NON_FRAUD_TRANSACTIONS + 1):
        person = random.choice(all_people)
        trn = person.transact()
        csv_writer.writerow(
            {
                "tid": tid,
                "src": trn.src,
                "dst": trn.dst,
                "amt": trn.amt,
                "time": trn.time,
                "day_of_week": trn.day_of_week,
                "is_fraud": trn.is_fraud,
            }
        )
    for tid in range(
        N_NON_FRAUD_TRANSACTIONS + 1,
        N_NON_FRAUD_TRANSACTIONS + 1 + N_FRAUD_TRANSACTIONS,
    ):
        fraudster_idx = random.choice(list(FRAUDSTER_IND))
        csv_writer.writerow(
            {
                "tid": tid,
                "src": random.choice(all_people).pid,
                "dst": all_people[fraudster_idx].pid,
                "amt": round(random.uniform(0, 1000), 2),
                "time": random.choice(range(1, 25)),
                "day_of_week": random.choice([1, 2, 3, 4, 5, 6, 7]),
                "is_fraud": True,
            }
        )

print(f"Simulated dataset written to '{SIM_DATA_OUTPUT_PATH}'")
