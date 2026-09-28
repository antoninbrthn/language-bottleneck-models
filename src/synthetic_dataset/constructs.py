from enum import Enum

class Construct(Enum):
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"

def compute_correct_answer(constructs, operands):
    """
    Compute the correct answer for a question that might involve one or more constructs.
    """
    if not constructs:
        return None

    # For now if there's more than one, take the first
    main_construct = constructs[0]

    if main_construct == Construct.ADDITION:
        return sum(operands)
    elif main_construct == Construct.SUBTRACTION:
        # For two operands
        return operands[0] - operands[1]
    elif main_construct == Construct.MULTIPLICATION:
        result = 1
        for op in operands:
            result *= op
        return result
    elif main_construct == Construct.DIVISION:
        if len(operands) != 2 or operands[1] == 0:
            return None 
        # division rounded to nearest integer
        return round(operands[0] / operands[1])
    else:
        return None
