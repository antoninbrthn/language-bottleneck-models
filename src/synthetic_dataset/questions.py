# questions.py
import numpy as np
from .constructs import Construct, compute_correct_answer
import pandas as pd

import numpy as np
from .constructs import Construct, compute_correct_answer

class Question:
    def __init__(self, question_id, constructs, operands):
        """
        :param question_id: unique identifier
        :param constructs: list of constructs (e.g. [Construct.ADDITION])
        :param operands: list of integers
        """
        self.question_id = question_id
        self.constructs = constructs
        self.operands = operands
        self.correct_answer = compute_correct_answer(self.constructs, self.operands)
        self.text = self.generate_question_text()

    def generate_question_text(self):
        operator_map = {
            Construct.ADDITION: '+',
            Construct.SUBTRACTION: '-',
            Construct.MULTIPLICATION: '*',
            Construct.DIVISION: '/'
        }
        operator = operator_map.get(self.constructs[0], '?')
        txt = f"What is {self.operands[0]} {operator} {self.operands[1]} "
        if operator == '/':
            txt += f" (rounded to the nearest integer)?"
        else:
            txt += "?"
        return txt

    def __str__(self):
        return (f"Question(qid={self.question_id}, "
                f"constructs={[c.value for c in self.constructs]}, "
                f"operands={self.operands}, "
                f"correct_answer={self.correct_answer})")
    
    def to_series(self):
        """Convert question to pandas Series for easy DataFrame creation"""
        return pd.Series({
            'question_id': self.question_id,
            'constructs': str([c.value for c in self.constructs]),  # Store construct values instead of objects
            'operands': str(self.operands),
            'correct_answer': self.correct_answer,
            'text': self.text
        })

    @classmethod
    def from_series(cls, series):
        """Create Question instance from pandas Series"""
        question_id = series['question_id']
        constructs_str = series['constructs']
        operands_str = series['operands']
        
        # Convert constructs string back to list of Construct objects
        construct_values = eval(constructs_str)
        constructs = [Construct(value) for value in construct_values]
        
        # Convert operands string back to list
        operands = eval(operands_str)
        
        return cls(
            question_id=question_id,
            constructs=constructs,
            operands=operands
        )

class QuestionGenerator:
    def __init__(self, n_questions, rng_seed=42):
        self.n_questions = n_questions
        self.rng = np.random.default_rng(rng_seed)
        self.all_constructs = [
            Construct.ADDITION,
            Construct.SUBTRACTION,
            Construct.MULTIPLICATION,
            Construct.DIVISION,
        ]

    def generate_questions(self):
        """
        Example approach: pick a construct at random, pick 2 random operands in [0..20],
        etc. Return a list of Question objects.
        """
        questions = []
        for qid in range(self.n_questions):
            c = self.rng.choice(self.all_constructs)
            if c in [Construct.ADDITION, Construct.SUBTRACTION]:
                op1 = self.rng.integers(0, 16) # 0 to 15
                op2 = self.rng.integers(0, 16)
            else:  # multiplication or division
                op1 = self.rng.integers(1, 11) # 1 to 10
                op2 = self.rng.integers(1, 11)  
                # order operands for division
                if c == Construct.DIVISION:
                    op1, op2 = max(op1, op2), min(op1, op2)

            constructs = [c]  # single-construct example
            q = Question(question_id=qid, constructs=constructs, operands=[op1, op2])
            questions.append(q)

        return questions
