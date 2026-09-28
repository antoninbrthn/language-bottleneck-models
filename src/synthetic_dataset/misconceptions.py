from abc import ABC, abstractmethod
import math
from .constructs import Construct

class MisconceptionRegistry:
    """Registry to automatically collect all misconception classes"""
    _misconceptions = {}

    @classmethod
    def register(cls, misconception_class):
        """Register a misconception class"""
        cls._misconceptions[misconception_class.__name__] = misconception_class
        return misconception_class

    @classmethod
    def get_misconception_class(cls, name):
        """Get a misconception class by name"""
        return cls._misconceptions.get(name)

    @classmethod
    def get_all_misconceptions(cls):
        """Get all registered misconception classes"""
        return cls._misconceptions

class Misconception(ABC):
    """
    Base class for a misconception. Each misconception:
      - Has a textual description.
      - Has parameters controlling how often or strongly it applies.
    """

    def __init__(self, description, base_prob=1.0, **kwargs):
        """
        :param description: A textual description of the misconception.
        :param base_prob: The base probability that this misconception will be triggered,
                          given that it is relevant. Defaults to 1.
        :param **kwargs: Additional parameters specific to each misconception type.
        """
        self.description = description
        self.base_prob = base_prob
        # Store any additional kwargs as instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses with the registry"""
        super().__init_subclass__(**kwargs)
        MisconceptionRegistry.register(cls)

    @abstractmethod
    def relevance(self, question):
        """
        Returns a number in [0, 1] indicating how relevant this misconception is
        for the given question. For example, forgetting to carry is relevant
        only if the question is multi-digit addition.
        """
        pass

    @abstractmethod
    def apply(self, question, correct_answer, rng):
        """
        Returns the student's final answer if the misconception is applied.
        Usually a 'distorted' or 'incorrect' version of correct_answer.
        """
        pass

    def __str__(self):
        return f"Misconception(description={self.description}, base_prob={self.base_prob})"

    def maybe_apply(self, question, correct_answer, rng):
        """
        Utility: combine the relevance with the base probability to decide
        if we actually apply the misconception. If not relevant or random draw
        fails, we just return the original correct answer.
        """
        rel = self.relevance(question)  # e.g. 0 or 1
        # Probability of applying = base_prob * relevance
        p_apply = rel * self.base_prob
        if rng.random() < p_apply:
            return self.apply(question, correct_answer, rng)
        return correct_answer

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of this misconception.
        """
        raise NotImplementedError

class ForgetToCarry(Misconception):
    """
    If the question is multi-digit addition, the student forgets to carry.
    For demonstration, we do a naive 'digit-wise sum mod 10' approach.
    """

    def relevance(self, question):
        # Check if addition is in constructs
        # Then check if multi-digit, etc.
        if Construct.ADDITION in question.constructs:
            # If sum of operands >= 10, relevant
            if sum(question.operands) >= 10:
                return 1.0
        return 0.0

    def apply(self, question, correct_answer, rng):
        # For simplicity, assume exactly 2 operands. 
        op1, op2 = question.operands
        max_len = max(len(str(op1)), len(str(op2)))
        str_op1 = str(op1).zfill(max_len)
        str_op2 = str(op2).zfill(max_len)
        result_digits = []
        for d1, d2 in zip(str_op1[::-1], str_op2[::-1]):
            partial_sum = int(d1) + int(d2)
            digit = partial_sum % 10
            result_digits.append(str(digit))
        mistaken_answer = int("".join(result_digits[::-1]))
        return mistaken_answer

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of ForgetToCarry.
        """
        return cls(description="Forgets to carry in addition (when a column sum >= 10)", base_prob=1.0)

# class FlawedMultiplication(Misconception):
#     """
#     Example: A student systematically believes certain multiplication
#     facts are some fixed wrong value. Let user provide a dictionary mapping
#     (x, y) -> wrong value.

#     E.g. { (3,3): 6, (7,8): 48 }
#     """

#     def __init__(self, description, base_prob, flawed_facts=None):
#         super().__init__(description, base_prob)
#         self.flawed_facts = flawed_facts if flawed_facts else {}

#     def relevance(self, question):
#         if Construct.MULTIPLICATION in question.constructs:
#             # If the question's operand pair is in our flawed_facts, relevant=1.0
#             # else 0
#             if len(question.operands) == 2:
#                 sorted_pair = tuple(sorted(question.operands))
#                 if sorted_pair in self.flawed_facts:
#                     return 1.0
#         return 0.0

#     def apply(self, question, correct_answer, rng):
#         sorted_pair = tuple(sorted(question.operands))
#         return self.flawed_facts.get(sorted_pair, correct_answer)

#     @classmethod
#     def random_instance(cls, rng):
#         """
#         Generate a random instance of FlawedMultiplication with 1-4 flawed facts.
#         """
#         flawed_facts = {
#             (op1 := rng.integers(6, 10), op2 := rng.integers(6, 10)): (op1 * op2) + rng.choice([-1, 1])
#             for _ in range(rng.integers(1, 4))
#         }
#         return cls(description="Has flawed multiplication facts", base_prob=1.0, flawed_facts=flawed_facts)

class FailsSpecificMultiplication(Misconception):
    """
    A student consistently fails multiplication when one of the operands is a specific number.
    For example, they always get multiplication wrong if one of the numbers is 7.
    """

    def __init__(self, description, base_prob, failed_number):
        super().__init__(description, base_prob)
        self.failed_number = failed_number

    def relevance(self, question):
        if Construct.MULTIPLICATION in question.constructs:
            if self.failed_number in question.operands:
                return 1.0
        return 0.0

    def apply(self, question, correct_answer, rng):
        # Ensure the wrong answer is slightly off the correct one
        return correct_answer + rng.choice([-1, 1])

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of FailsSpecificMultiplication.
        """
        failed_number = rng.integers(6, 10)  # Pick a random number between 6 and 9
        return cls(
            description=f"Fails multiplication involving {failed_number} as an operand",
            base_prob=1.0,
            failed_number=failed_number
        )

class FailsSpecificNumber(Misconception):
    """
    A student consistently fails any construct when one of the operands is a specific number.
    """

    def __init__(self, description, base_prob, failed_number):
        super().__init__(description, base_prob)
        self.failed_number = failed_number

    def relevance(self, question):
        if self.failed_number in question.operands:
            return 1.0
        return 0.0

    def apply(self, question, correct_answer, rng):
        # Ensure the wrong answer is slightly off the correct one
        return correct_answer + rng.choice([-1, 1])

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of FailsSpecificMultiplication.
        """
        failed_number = rng.integers(6, 10)  # Pick a random number between 6 and 9
        return cls(
            description=f"Fails any operation involving {failed_number} as an operand",
            base_prob=1.0,
            failed_number=failed_number
        )

class FailsForBigNumbers(Misconception):
    """
    Student fails for any question if an operand > threshold.
    We can define what "fail" means: either random wrong guess or a fixed sentinel value.
    """
    def __init__(self, description, base_prob=1.0, threshold=10, fail_delta=1e6):
        super().__init__(description, base_prob)
        self.threshold = threshold
        self.fail_delta = fail_delta

    def relevance(self, question):
        # If any operand > threshold, relevant
        if any(op > self.threshold for op in question.operands):
            return 1.0
        return 0.0

    def apply(self, question, correct_answer, rng):
        return correct_answer + self.fail_delta

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of FailsForBigNumbers with a random threshold.
        """
        threshold = rng.choice([10])  # Randomly pick a threshold for "big numbers"
        # fail_delta = rng.choice([-1, 1])  # Small random penalty for failure
        fail_delta = 1e6
        return cls(
            description=f"Fails any operation with operands > {threshold}",
            base_prob=1.0,
            threshold=threshold,
            fail_delta=fail_delta
        )

class RoundDownDivision(Misconception):
    """
    If the question is division, the student incorrectly always rounds down.
    """
    def relevance(self, question):
        return 1.0 if Construct.DIVISION in question.constructs else 0.0

    def apply(self, question, correct_answer, rng):
        if Construct.DIVISION not in question.constructs:
            return correct_answer
        if len(question.constructs) > 1:
            raise Exception("WARNING: RoundDownDivision only implemeneted for single construct division questions.")
        # Compute actual division result and round down
        dividend, divisor = question.operands
        actual_result = dividend / divisor
        return math.floor(actual_result)

    @classmethod
    def random_instance(cls, rng):
        """
        Generate a random instance of RoundDownDivision.
        """
        return cls(description="Always rounds down division results", base_prob=1.0)

class StruggleWithNegativeNumbers(Misconception):
    """
    If the correct answer is negative, the student gets it wrong.
    """
    def relevance(self, question):
        return 1.0 if question.correct_answer < 0 else 0.0

    def apply(self, question, correct_answer, rng):
        return correct_answer + 1e6 if correct_answer < 0 else correct_answer

    @classmethod
    def random_instance(cls, rng):
        """ Generate a random instance of StruggleWithNegativeNumbers. """
        return cls(description="Fails with negative numbers", base_prob=1.0)
