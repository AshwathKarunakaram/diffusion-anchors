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
    # --- Arithmetic-finale family -------------------------------------------
    # The first sweep showed the mixed correct/locked-wrong zone is prompts
    # whose derivation is easy but whose final answer needs a multi-digit
    # computation (two_aces_hand: 6 * 17,296).  These clones target that zone.
    LockinPrompt(
        "one_ace_hand",
        "How many 5-card hands from a standard 52-card deck contain exactly one ace?",
        778320,  # C(4,1) * C(48,4) = 4 * 194,580
    ),
    LockinPrompt(
        "two_hearts_hand",
        "How many 5-card hands from a standard 52-card deck contain exactly two hearts?",
        712842,  # C(13,2) * C(39,3) = 78 * 9,139
    ),
    LockinPrompt(
        "distinct_letter_string",
        "How many 5-letter strings over the 26-letter English alphabet have all letters distinct?",
        7893600,  # 26 * 25 * 24 * 23 * 22
    ),
    LockinPrompt(
        "committee_boys_girls",
        "A committee is formed by choosing 4 of 15 boys and 3 of 12 girls. How many committees are possible?",
        300300,  # C(15,4) * C(12,3) = 1,365 * 220
    ),
    LockinPrompt(
        "binary_choose_eight",
        "How many binary strings of length 20 contain exactly eight 1s?",
        125970,  # C(20,8)
    ),
    LockinPrompt(
        "sum_to_999",
        "What is the sum of all integers from 1 through 999 inclusive?",
        499500,  # 999 * 1000 / 2
    ),
    LockinPrompt(
        "handshakes_150",
        "At a party of 150 people, every pair of people shakes hands exactly once. How many handshakes occur?",
        11175,  # C(150,2) = 150 * 149 / 2
    ),
)
