import csv
import os
import pandas as pd
import argparse

from scipy import misc

from src.utils.config import DATA_DIR, PROJECT_DATA_DIR
from src.synthetic_dataset.students import Student
from src.synthetic_dataset.questions import Question, QuestionGenerator
from src.synthetic_dataset.misconceptions import *
from src.synthetic_dataset.constructs import Construct
import numpy as np

def generate_students(n_students=5, rng_seed=100):
    """
    Create n_students, each with random skill mastery among {0, 0.5, 1.0}.
    Also attach a random subset of misconceptions, each generated with appropriate parameters.
    """
    rng = np.random.default_rng(rng_seed)
    
    # Potential skill levels for each construct
    # possible_levels = [0.0, 0.5, 1.0]
    possible_levels = [0.0, 1.0]  # for now only 0 and 1

    # List of misconception classes that can be instantiated randomly
    misconception_classes = [
        ForgetToCarry,
        FailsForBigNumbers,
        RoundDownDivision,
        FailsSpecificMultiplication,
        FailsSpecificMultiplication,  # Duplicate to increase probability
        FailsSpecificNumber,
        # FailsSpecificNumber,  # Duplicate to increase probability
        StruggleWithNegativeNumbers,  # Duplicate to increase probability
    ]

    students = []
    for sid in range(n_students):
        # 3/4 chance to get 1.0, 1/4 chance to get 0.0
        skill_mastery = {c: rng.choice([0.0, 1.0], p=[0.25, 0.75]) for c in Construct}
        
        
        
        # Determine number of misconceptions with decreasing probability
        if rng.random() < 0.1:
            n_mis = 0
        else:
            # probs = np.array([0.95**(i+1) for i in range(len(misconception_classes))])
            # probs /= probs.sum()  # Normalize to make it a valid probability distribution
            # more likely to have 3-4 misconceptions
            probs = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.05]
            probs = np.array(probs + [0] * (len(misconception_classes) - len(probs)))
            probs /= probs.sum()
            n_mis = rng.choice(np.arange(1, len(misconception_classes) + 1), p=probs)

        # Pick and instantiate misconceptions
        chosen_classes = rng.choice(misconception_classes, size=n_mis, replace=False).tolist()
        chosen_misconceptions = [cls.random_instance(rng) for cls in chosen_classes]

        student = Student(
            student_id=sid,
            skill_mastery=skill_mastery,
            misconceptions=chosen_misconceptions,
            rng_seed=rng.integers(1, 9999)
        )
        students.append(student)
    
    return students



def generate_dataset(
        n_students=5,
        n_questions=20,
        trajectory_length=10,
        output_bool=False,
        output_path=None,
        rng_seed=999):
    """
    1) Generate students
    2) Generate questions
    3) For each student, sample some questions, record the answer
    4) Write to CSV
    """
    # 1) create students
    students = generate_students(n_students=n_students, rng_seed=rng_seed)

    # 2) create questions
    qg = QuestionGenerator(n_questions=n_questions, rng_seed=rng_seed+1)
    questions = qg.generate_questions()

    # 3) For each student, pick random questions
    rng = np.random.default_rng(rng_seed+2)
    data_rows = []
    for student in students:
        sampled_question_ids = rng.choice(range(n_questions), size=trajectory_length, replace=True)
        for qid in sampled_question_ids:
            q = questions[qid]
            ans, is_correct, misconceptions_applied = student.answer_question(q)
            data_rows.append({
                "student_id": student.student_id,
                "question_id": q.question_id,
                "constructs": [c.value for c in q.constructs],
                # df_processed["construct_id"] = df["constructs"].apply(lambda x: list(Construct).index(Construct(eval(x)[0])))
                "constructs_ids": [list(Construct).index(c) for c in q.constructs],
                "operands": q.operands,
                "correct_answer": q.correct_answer,
                "student_answer": ans,
                "is_correct": is_correct,
                "misconceptions_applied": misconceptions_applied,
            })

    # 4) write CSV
    if output_bool:
        # Split the output path
        base_path = output_path
        
        # Export each component
        export_students_csv(students, os.path.join(output_path, "students.csv"))
        export_questions_csv(questions, os.path.join(output_path, "questions.csv"))
        export_interactions_csv(data_rows, os.path.join(output_path, "data.csv"))

    # Optionally return (students, questions, data_rows)
    return students, questions, pd.DataFrame(data_rows)


def import_dataset(csv_path='data.csv'):
    """
    Re-import the CSV into a list of dictionaries.
    Convert fields as appropriate.
    """
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for line in reader:
            row = dict(line)
            # parse constructs, operands from string
            row["constructs"] = [Construct(c) for c in eval(row["constructs"])]
            row["operands"] = eval(row["operands"])
            row["is_correct"] = (row["is_correct"] == 'True')
            
            # parse correct_answer, student_answer
            # might be floats or ints or None
            try:
                val = float(row["correct_answer"])
                # if it's an integer float like '12.0', convert to int
                if val.is_integer():
                    val = int(val)
                row["correct_answer"] = val
            except:
                row["correct_answer"] = None

            try:
                val = float(row["student_answer"])
                if val.is_integer():
                    val = int(val)
                row["student_answer"] = val
            except:
                row["student_answer"] = None

            rows.append(row)
    return pd.DataFrame(rows)


def export_students_csv(students, output_path):
    """Export students to CSV file"""
    df = pd.DataFrame([s.to_series() for s in students])
    df.to_csv(output_path, index=False)
    print(f"Students data written to {os.path.abspath(output_path)}")
    return df

def export_questions_csv(questions, output_path):
    """Export questions to CSV file"""
    df = pd.DataFrame([q.to_series() for q in questions])
    df.to_csv(output_path, index=False)
    print(f"Questions data written to {os.path.abspath(output_path)}")
    return df

def export_interactions_csv(data_rows, output_path):
    """Export student-question interactions to CSV file"""
    df = pd.DataFrame(data_rows)
    # Convert constructs to their string values
    df['constructs'] = df['constructs'].apply(lambda x: str([c for c in x]))
    df['operands'] = df['operands'].apply(str)
    df.to_csv(output_path, index=False)
    print(f"Interactions data written to {os.path.abspath(output_path)}")
    return df

def import_dataset_full(base_path):
    """
    Import complete dataset including students, questions and interactions
    Args:
        base_path: Base path without extension, e.g. 'data/synthetic-11/synthetic_dataset'
    Returns:
        tuple: (students, questions, interactions)
    """
    # Import students
    students_df = pd.read_csv(os.path.join(base_path, "students.csv"))
    # students = [Student.from_series(row) for _, row in students_df.iterrows()]
    
    # Import questions
    questions_df = pd.read_csv(os.path.join(base_path, "questions.csv"))
    # questions = [Question.from_series(row) for _, row in questions_df.iterrows()]
    
    # Import interactions
    interactions_df = import_dataset(os.path.join(base_path, "data.csv"))
    
    return students_df, questions_df, interactions_df

def load_classes_from_csv(students_df, questions_df):
    """
    Load students and questions from CSV files
    Args:
        students_df: DataFrame with student data
        questions_df: DataFrame with question data
    Returns:
        tuple: (students, questions)
    """
    students = [Student.from_series(row) for _, row in students_df.iterrows()]
    questions = [Question.from_series(row) for _, row in questions_df.iterrows()]
    return students, questions



def main():
    parser = argparse.ArgumentParser(description='Generate synthetic datasets.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset (e.g., synthetic-11)', default='synthetic-11')
    parser.add_argument('--n_students', type=int, default=5, help='Number of students to generate')
    parser.add_argument('--n_questions', type=int, default=20, help='Number of questions to generate')
    parser.add_argument('--trajectory_length', type=int, default=10, help='Length of the trajectory')
    parser.add_argument('--output_bool', type=bool, default=False, help='Whether to output CSV')
    
    args = parser.parse_args()

    # Update output path based on dataset name
    output_path = os.path.join(PROJECT_DATA_DIR, args.dataset_name+'/')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output path: {output_path}")
    

    # Generate dataset with provided parameters
    students, questions, data = generate_dataset(
        n_students=args.n_students,
        n_questions=args.n_questions,
        trajectory_length=args.trajectory_length,
        output_bool=args.output_bool,
        output_path=output_path
    )

    if args.output_bool:
        # export arguments into txt file to reproduce command easily, with syntax --arg1 value1 --arg2 value2
        with open(os.path.join(os.path.dirname(output_path), "arguments.txt"), 'w') as f:
            for arg in vars(args):
                f.write(f"--{arg} {getattr(args, arg)}\n")


    # Show some sample knowledge summary for a student
    print(students[0].knowledge_summary())
    
    # dt = pd.DataFrame(data_rows)
    # dt.question_id.value_counts()
    # dt.is_correct.mean()
    # dt.groupby('student_id').is_correct.mean()
    print('Dataset info:')
    print(f'Number of students: {args.n_students}')
    print(f'Number of questions: {args.n_questions}')
    print(f'Average accuracy per questions: {data.groupby("question_id").is_correct.mean().mean()} +/- {data.groupby("question_id").is_correct.mean().std()}')
    print(f'Average accuracy per student: {data.groupby("student_id").is_correct.mean().mean()} +/- {data.groupby("student_id").is_correct.mean().std()}')
    # data.misconceptions_applied.mean()
    # incorrect.misconceptions_applied.mean()
    incorrect = data[~data.is_correct]
    print(f'Average misconceptions applied: {data.misconceptions_applied.mean()}')
    print(f'Average misconceptions applied for incorrect answers: {incorrect.misconceptions_applied.mean()}')
    # number of misc per student
    misc_per_student = [len(s.misconceptions) for s in students]
    print(f'Average misconceptions per student: {np.mean(misc_per_student)} +/- {np.std(misc_per_student)}')



    # Import
    imported = import_dataset(csv_path=os.path.join(output_path, 'data.csv'))
    students_df, questions_df, interactions_df = import_dataset_full(base_path=output_path)
    print("First 3 rows from imported dataset:")
    print(imported.head(3))

    
    # s = Student.from_series(students_df.iloc[0])
    # q = Question.from_series(questions_df.iloc[0])
    # students, questions = load_classes_from_csv(students_df, questions_df)
    # for i in range(len(interactions_df)):
    #     inter = interactions_df.iloc[i]
    #     r_s = students_df[students_df.student_id == int(inter.student_id)].iloc[0]
    #     s = Student.from_series(r_s)
    #     q_s = questions_df[questions_df.question_id == int(inter.question_id)].iloc[0]
    #     q = Question.from_series(q_s)
    #     # q.generate_question_text()
    #     # print(s.knowledge_summary())
    #     # check that loading went well
    #     assert s.answer_question(q)[1] == inter.is_correct

    prof_count = []
    for i, s in students_df.iterrows():
        prof_count.append(s.skill_mastery.count("1"))
    np.mean(prof_count)
    np.std(prof_count)
    # value counts
    pd.Series(prof_count).value_counts()


if __name__ == "__main__":
    main()

    # python src/synthetic_dataset/dataset_generator.py synthetic-11 --n_students 10 --n_questions 30 --trajectory_length 15 --output_bool True
