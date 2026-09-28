class PromptGenerator:
    # add an optional contains attribute
    def __init__(self, contains=None):
        self.contains = contains

    def create_prompt(self, trajectory, new_question):
        history = []
        history.append("The student answered the following questions:\n")
        for entry in trajectory:
            correctness = "correctly" if entry["correct"] == 1 else "incorrectly"
            question_txt = format_question(entry, self.contains)
            construct_txt = format_construct(entry, self.contains)
            if len(question_txt)==0:
                history.append(f"Question with construct {construct_txt}: answered {correctness}.\n")
            elif len(construct_txt)==0:
                history.append(f"Question {question_txt}: answered {correctness}.\n")
            else:
                history.append(f"Question {question_txt}, with construct {construct_txt}: answered {correctness}.\n")

        # New question formatting
        new_question_txt = format_question(new_question, self.contains)  # Assuming new_question is both ID and text

        prompt = " ".join(history) + \
                 f"Predict whether the student will answer {new_question_txt} correctly or not. " + \
                 "Answer with \"Yes\" or \"No\" and nothing else."
        return prompt

    def create_prompt_bottleneck(self, trajectory, new_question):
        history = []
        history.append("The student answered the following questions:\n")
        for entry in trajectory:
            correctness = "correctly" if entry["correct"] == 1 else "incorrectly"
            question_txt = format_question(entry, self.contains)
            construct_txt = format_construct(entry, self.contains)
            if len(question_txt)==0:
                history.append(f"Question with construct {construct_txt}: answered {correctness}.\n")
            elif len(construct_txt)==0:
                history.append(f"Question {question_txt}: answered {correctness}.\n")
            else:
                history.append(f"Question {question_txt}, with construct {construct_txt}: answered {correctness}.\n")
        
        # New question formatting
        new_question_txt = format_question(new_question, self.contains)  # Assuming new_question is both ID and text

        x_prompt = " ".join(history)
        y_prompt = f"Predict whether the student will answer {new_question_txt} correctly or not. " + \
                 "Answer with \"Yes\" or \"No\" and nothing else."
        return x_prompt, y_prompt

    def create_prompt_bottleneck_multiple_new_qs(self, trajectory, new_questions):
        history = []
        history.append("The student answered the following questions:\n")
        for entry in trajectory:
            correctness = "correctly" if entry["correct"] == 1 else "incorrectly"
            question_txt = format_question(entry, self.contains)
            construct_txt = format_construct(entry, self.contains)
            if len(question_txt)==0:
                history.append(f"Question with construct {construct_txt}: answered {correctness}.\n")
            elif len(construct_txt)==0:
                history.append(f"Question {question_txt}: answered {correctness}.\n")            
            else:
                history.append(f"Question {question_txt}, with construct {construct_txt}: answered {correctness}.\n")
        
        # New question formatting
        y_prompts = []
        for new_question in new_questions:
            new_question_txt = format_question(new_question, self.contains)  # Assuming new_question is both ID and text

            x_prompt = " ".join(history)
            y_prompt = f"Predict whether the student will answer {new_question_txt} correctly or not. " + \
                    "Answer with \"Yes\" or \"No\" and nothing else."
            y_prompts.append(y_prompt)
        return x_prompt, y_prompts
    
    def create_prompt_multiple_new_qs(self, trajectory, new_questions):
        history = []
        history.append("The student answered the following questions:\n")
        for entry in trajectory:
            correctness = "correctly" if entry["correct"] == 1 else "incorrectly"
            question_txt = format_question(entry, self.contains)
            construct_txt = format_construct(entry, self.contains)
            if len(question_txt)==0:
                history.append(f"Question with construct {construct_txt}: answered {correctness}.\n")
            elif len(construct_txt)==0:
                history.append(f"Question {question_txt}: answered {correctness}.\n")
            else:
                history.append(f"Question {question_txt}, with construct {construct_txt}: answered {correctness}.\n")

        # New question formatting
        prompts = []
        for new_question in new_questions:
            new_question_txt = format_question(new_question, self.contains)  # Assuming new_question is both ID and text
            prompt = " ".join(history) + \
                    f"Predict whether the student will answer {new_question_txt} correctly or not. " + \
                    "Answer with \"Yes\" or \"No\" and nothing else."
            prompts.append(prompt)
        return prompts

    @staticmethod
    def parse_response(response):
        response = response.strip().lower()
        return "Yes" in response

def format_question(entry, contains):
    q_id = entry.get('question_id', None)
    q_text = entry.get('question_text', None)
    if "question_id" in contains and "question_text" in contains:
        return f"ID {q_id} ({q_text})"
    elif "question_id" in contains:
        return f"ID {q_id}"
    elif "question_text" in contains:
        return f"{q_text}"
    return ""


def format_construct(entry, contains):
    construct_id = entry.get('construct_id', None)
    construct_text = entry.get('construct_text', None)
    if "construct_id" in contains and "construct_text" in contains:
        return f"ID {construct_id} ({construct_text})"
    elif "construct_id" in contains:
        return f"ID {construct_id}"
    elif "construct_text" in contains:
        return f"{construct_text}"
    return ""