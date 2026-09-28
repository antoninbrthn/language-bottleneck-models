# students.py
import numpy as np
from src.synthetic_dataset.constructs import Construct
# import all misconceptions
from src.synthetic_dataset.misconceptions import MisconceptionRegistry
from src.synthetic_dataset.misconceptions import *
import pandas as pd

class Student:
    def __init__(self, student_id, skill_mastery, misconceptions=None, rng_seed=123):
        """
        :param student_id: Unique identifier
        :param skill_mastery: dict(Construct -> float) in {0, 0.5, 1.0}
        :param misconceptions: list of Misconception objects
        :param rng_seed: seed for reproducible random draws
        """
        self.student_id = student_id
        self.skill_mastery = skill_mastery
        self.misconceptions = misconceptions if misconceptions else []
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
    
    def knowledge_summary(self):
        """
        Return a textual description of the student's skill levels and misconceptions.
        """
        mastered_skills = []
        non_mastered_skills = []
        for construct, mastery in self.skill_mastery.items():
            if mastery == 1.0:
                mastered_skills.append(construct.value)
            elif mastery == 0.0:
                non_mastered_skills.append(construct.value)

        mis_strs = []
        for m in self.misconceptions:
            mis_strs.append(f"{m.description}")

        # Template:
        # """The student has mastered the following skill(s):
        # - <skill 1>
        # - <skill 2>
        # but have not mastered the following skill(s): 
        # - <skill 3>
        # - <skill 4>
        # They have the following misconceptions:
        # - <misconception 1 (txt)>
        # - <misconception 2 (txt)>
        # - ...
        # """

        if mastered_skills:
            summary = "The student always gets the following constructs correct (unless any of their misconceptions apply):\n"
            summary += "\n".join([f"- {skill}" for skill in mastered_skills])
        else:
            summary = "The student has not mastered any skills.\n"
        summary += "\n"
        if non_mastered_skills:
            summary += "The student always fails on the following constructs:\n"
            summary += "\n".join([f"- {skill}" for skill in non_mastered_skills])
        else:
            summary += "The student has mastered all skills.\n"
        summary += "\n"
        if mis_strs:
            summary += "They have the following systematic misconceptions:\n"
            summary += "\n".join([f"- {mis}" for mis in mis_strs])
        else:
            summary += "They have no misconceptions.\n"
        # summary += "\n"
        # summary += "Behavior:\n"
        # summary += "- Answers incorrectly if the construct is not mastered, or if one of their misconception applies.\n"
        # summary += "- Answers correctly if the construct is mastered and no misconception applies.\n"
        return summary

    def answer_question(self, question):
        """
        Returns (answer, is_correct).
        Probability logic:
          1) base_probability = aggregator of skill for all constructs in question
          2) do a random draw: if pass => student 'knows' how to do it
             if fail => produce a random guess in some range
          3) if the student 'knows' it, we apply misconceptions in sequence. 
        """
        # 1) Combine skill levels. For simplicity, use min mastery across constructs
        #    so that if a question has multiple constructs, a single weak spot dooms them.
        if not question.constructs:
            base_probability = 0.0
        else:
            levels = [self.skill_mastery.get(c, 0.0) for c in question.constructs]
            base_probability = min(levels)

        # 2) random draw
        if self.rng.random() > base_probability:
            # Student does not produce correct answer. 
            # Answer is `delta` away from the correct answer, but not 0.
            delta = self.rng.choice(np.arange(-5, 6))
            # Ensure delta is not 0
            while delta == 0:
                delta = self.rng.choice(np.arange(-5, 6))
            answer = question.correct_answer + delta
            is_correct = (answer == question.correct_answer)
            misconceptions_applied = False
            return answer, is_correct, misconceptions_applied
        
        # If the student "knows" it, they start with the correct answer
        answer = question.correct_answer

        pre_misc_is_correct = get_is_correct(question.correct_answer, answer)
        # 3) Now apply misconceptions in sequence
        for mis in self.misconceptions:
            answer = mis.maybe_apply(question, answer, self.rng)

        # We define correctness after all misconceptions have possibly changed the answer
        is_correct = get_is_correct(question.correct_answer, answer)
        
        misconceptions_applied = is_correct != pre_misc_is_correct

        return answer, is_correct, misconceptions_applied

    def to_series(self):
        """Convert student to pandas Series for easy DataFrame creation"""
        # Convert skill mastery to a more concrete format
        skill_mastery_dict = {}
        for construct, level in self.skill_mastery.items():
            skill_mastery_dict[construct.value] = level  # Use construct value instead of object
            
        # Convert misconceptions to concrete format
        misconceptions_list = []
        for m in self.misconceptions:
            mis_dict = {
                'type': m.__class__.__name__,
                'description': m.description,
                'base_prob': m.base_prob
            }
            # Add any additional parameters specific to each misconception type
            if hasattr(m, 'failed_number'):
                mis_dict['failed_number'] = m.failed_number
            if hasattr(m, 'threshold'):
                mis_dict['threshold'] = m.threshold
            if hasattr(m, 'fail_delta'):
                mis_dict['fail_delta'] = m.fail_delta
            misconceptions_list.append(mis_dict)
            
        return pd.Series({
            'student_id': self.student_id,
            'skill_mastery': str(skill_mastery_dict),  # Store as string of dict
            'misconceptions': str(misconceptions_list),  # Store as string of list
            'rng_seed': self.rng_seed,
            'knowledge_state': self.knowledge_summary()
        })

    @classmethod
    def from_series(cls, series):
        """Create Student instance from pandas Series"""
        student_id = series['student_id']
        skill_mastery_str = series['skill_mastery']
        misconceptions_str = series['misconceptions']
        rng_seed = series['rng_seed']
        
        # Convert skill mastery string back to dict
        skill_mastery_dict = eval(skill_mastery_str)
        skill_mastery = {}
        for construct_value, level in skill_mastery_dict.items():
            construct = Construct(construct_value)  # Convert string to Construct
            skill_mastery[construct] = level
            
        # Convert misconceptions string back to list of dicts
        misconceptions_list = eval(misconceptions_str)
        misconceptions = []
        for mis_dict in misconceptions_list:
            mis_type = mis_dict['type']
            # Get the misconception class from the registry
            mis_class = MisconceptionRegistry.get_misconception_class(mis_type)
            if mis_class is None:
                raise ValueError(f"Unknown misconception type: {mis_type}")
            
            # Create kwargs dict excluding 'type' since it's not a constructor parameter
            kwargs = {k: v for k, v in mis_dict.items() if k != 'type'}
            
            # Create the misconception instance with the kwargs
            misconceptions.append(mis_class(**kwargs))
        student = cls(
            student_id=student_id,
            skill_mastery=skill_mastery,
            misconceptions=misconceptions,
            rng_seed=rng_seed
        )

        assert series['knowledge_state'] == student.knowledge_summary(), "Knowledge state does not match the original series"
        

        return student

def is_close(a, b):
    return abs(a - b) < 1e-6

def get_is_correct(correct_answer, answer):
    if isinstance(correct_answer, float):
        return is_close(correct_answer, answer)
    return correct_answer == answer
