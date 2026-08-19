"""Short answer-first quantitative prompts for the lock-in replication screen.

All answers are integers so the first visible numerical answer can be traced
without an LLM judge.  Prompts are deliberately short enough that the answer
and reasoning should fit in one 256-token canvas.
"""

from dataclasses import dataclass


ANSWER_FIRST_SUFFIX = "State your answer first, then give your reasoning."


@dataclass(frozen=True)
class LockinPrompt:
    name: str
    question: str
    answer: int

    @property
    def prompt(self) -> str:
        return f"{self.question} {ANSWER_FIRST_SUFFIX}"


PROMPTS = (
    LockinPrompt(
        "squares_400_800",
        "How many square numbers are there between 400 and 800?",
        8,
    ),
    LockinPrompt(
        "multiples_union",
        "How many positive integers at most 100 are divisible by 6 or by 10?",
        22,
    ),
    LockinPrompt(
        "choose_three",
        "How many ways are there to choose 3 people from a group of 8 people?",
        56,
    ),
    LockinPrompt(
        "divisors_360",
        "How many positive divisors does 360 have?",
        24,
    ),
    LockinPrompt(
        "trailing_zeros",
        "How many trailing zeros are in the decimal representation of 100 factorial?",
        24,
    ),
    LockinPrompt(
        "anagrams_level",
        "How many distinct arrangements are there of the letters in LEVEL?",
        30,
    ),
    LockinPrompt(
        "nonnegative_solutions",
        "How many nonnegative integer solutions are there to a + b + c = 10?",
        66,
    ),
    LockinPrompt(
        "primes_to_100",
        "How many integers from 1 through 100 inclusive have exactly two positive divisors?",
        25,
    ),
    LockinPrompt(
        "four_digit_two_zeros",
        "How many length-4 strings of decimal digits contain exactly two zeros?",
        486,
    ),
    LockinPrompt(
        "binary_choose_four",
        "How many binary strings of length 10 contain exactly four 1s?",
        210,
    ),
    LockinPrompt(
        "increasing_three_digits",
        "How many three-digit positive integers have digits that are strictly increasing from left to right?",
        84,
    ),
    LockinPrompt(
        "two_aces_hand",
        "How many 5-card hands from a standard 52-card deck contain exactly two aces?",
        103776,
    ),
)
