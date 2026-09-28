import re
import numpy as np
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import wandb
from transformers import TrainerCallback
import random
from src.models.lb_models.utils import add_instructions_to_xprompt, add_instructions_to_yprompt
from src.evaluation.evaluate_model import ModelEvaluator
from src.utils.config import LOGS_DIR, RESULTS_DIR
from src.models.hf_models import HFModel
from src.synthetic_dataset.constructs import Construct


# Utility functions for student info processing
def get_student_misc(student_dict, user_id, n_misc=1):
    """Extract n random misconception texts from the student.

    Args:
        student_dict: Dictionary of student objects {user_id: Student}
        user_id: The student ID
        n_misc: Number of misconceptions to extract

    Returns:
        str: Misconception descriptions or empty string if student has none
    """
    try:
        student = student_dict.get(int(user_id))
        if not student or not student.misconceptions:
            return ""

        # Pick n random misconceptions without replacement
        random_misconceptions = random.sample(student.misconceptions, min(n_misc, len(student.misconceptions)))
        # Join them into a single string
        misc_str = "\n".join([f"{misc.description}" for misc in random_misconceptions])
        return misc_str
    except Exception as e:
        print(f"Error getting misconceptions for user {user_id}: {e}")
        return ""


def get_student_mastery(student_dict, user_id, construct_id):
    """Extract whether the student has mastered a particular construct.

    Args:
        student_dict: Dictionary of student objects {user_id: Student}
        user_id: The student ID
        construct_id: The ID of the construct to check mastery for

    Returns:
        str: Description of the student's mastery level for the construct
    """
    try:
        student = student_dict.get(int(user_id))
        if not student:
            return ""

        # Convert construct_id to Construct enum
        construct_id = int(construct_id)
        construct = list(Construct)[construct_id]
        mastery = student.skill_mastery.get(construct, 0.0)

        if mastery == 1.0:
            return f"The student has mastered {construct.value} except in the event of misconceptions."
        else:
            return f"The student always fails with {construct.value}."
    except (ValueError, IndexError, Exception) as e:
        print(f"Error getting mastery for user {user_id}, construct {construct_id}: {e}")
        return ""


def process_dynamic_info(info_str, student_dict, user_id):
    """Process dynamic information strings like $get_student_misc:1 or $get_student_mastery:3

    Args:
        info_str: The information string that may contain dynamic elements
        student_dict: Dictionary of student objects {user_id: Student}
        user_id: The user ID to use for this processing

    Returns:
        str: The processed information string
    """
    if not info_str or not isinstance(info_str, str):
        return info_str

    # Check for dynamic elements
    if "$get_student_misc:" in info_str:
        # Extract the number of misconceptions to get
        parts = info_str.split("$get_student_misc:")
        n_misc = int(parts[1].split()[-1])
        misc_info = get_student_misc(student_dict, user_id, n_misc)
        if misc_info:
            return f"The student has at least the following misconception(s):\n{misc_info}"
        else:
            return "The student has no known misconceptions."

    elif "$get_student_mastery:" in info_str:
        # Extract the construct ID from the string
        parts = info_str.split("$get_student_mastery:")
        construct_id = int(parts[1].split()[-1])
        if construct_id is not None:
            return get_student_mastery(student_dict, user_id, construct_id)
        else:
            return ""

    return info_str


class GRPODataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_prompt_length=512,
        max_bottleneck_tokens=512,
        use_cot=False,
        conversational=True,
        x_template=None,
        additional_x_info=None,
        student_dict=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.conversational = conversational
        self.use_cot = use_cot
        self.max_bottleneck_tokens = max_bottleneck_tokens
        self.x_template = x_template
        self.additional_x_info = additional_x_info
        self.student_dict = student_dict
        # Convert dictionary to list of (key, value) pairs for indexing
        self.data_items = list(data.items())

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        user_id, ((x_prompt, y_prompt), label) = self.data_items[idx]

        # Process additional_x_info if provided and we have a student_dict
        additional_x_info = self.additional_x_info
        if additional_x_info and self.student_dict:
            additional_x_info = process_dynamic_info(additional_x_info, self.student_dict, user_id)

        if additional_x_info:
            x_prompt = f"{x_prompt}\n{additional_x_info}"

        x_prompt = add_instructions_to_xprompt(
            x_prompt, use_cot=self.use_cot, max_bottleneck_tokens=self.max_bottleneck_tokens, template=self.x_template
        )

        # Print warning if longer than max_length
        prompt_token_length = np.array(self.tokenizer(x_prompt)["input_ids"]).shape[-1]
        if prompt_token_length > self.max_prompt_length:
            print(f"Warning: input sequence of length {prompt_token_length} longer than max_length: {self.max_prompt_length}. Will be truncated.")

        # Encode input sequence
        encoding = self.tokenizer(x_prompt, truncation=True, max_length=self.max_prompt_length, padding="max_length", return_tensors="pt")

        # decode back to text
        x_prompt = self.tokenizer.decode(encoding["input_ids"].squeeze())

        if self.conversational:
            return {
                "prompt": [{"role": "user", "content": x_prompt}],
                "y_prompts": y_prompt,  # next question formatted in reward function so no need for conversation format
                "labels": label,
                "user_id": user_id,
            }
        else:
            return {"prompt": x_prompt, "y_prompts": y_prompt, "labels": label, "user_id": user_id}


def create_reward_function(decoder_model, decoder_batch_size=None, y_template=None):
    print(f"Using decoder_batch_size {decoder_batch_size}")

    def reward_func(prompts, completions, labels, y_prompts, **kwargs):
        """
        Compute rewards: given the bottlenecks produced by the encoder (`completions`), use the decoder to predict the answer to the question(s) (`y_prompts`) and compare it to the true answer(s) (`labels`).
        - All inputs should be repeats of length `num_generations`.
        - Each `labels` and `y_prompts` are either a single question or a list of questions.
        - If `y_prompts` is a list of questions, the reward is the average of the rewards for each question.
        - If `y_prompts` is a single question, the reward is 0 or 1.
        """
        full_print = np.random.rand() < 0.05
        if full_print:
            print(">>> num_generations:", len(prompts))
            prompt_contents = [p[0]["content"] if type(p) == list else p for p in prompts]
            nb_unique_prompts = len(set(prompt_contents))
            print(">>> num_unique_prompts:", nb_unique_prompts)
            print(">>> completion:", completions[0])
            print(">>> labels:", labels[0])

        # Normalize inputs and collect all decoder inputs
        all_decoder_inputs = []
        counts = []
        labels_flat = []
        for bottleneck, y_prompt, label in zip(completions, y_prompts, labels):
            # Normalize to list
            if not isinstance(y_prompt, list):
                y_prompt = [y_prompt]
                label = [label]
            inputs = [add_instructions_to_yprompt(y, bottleneck[0]["content"], template=y_template) for y in y_prompt]
            all_decoder_inputs.extend(inputs)
            counts.append(len(inputs))
            labels_flat.extend(label)

        # Call decoder once
        try:
            # If decoder_batch_size is not None, process inputs by batches
            if decoder_batch_size is not None:
                print("Decoding in batches of size", decoder_batch_size)
                all_decoder_outputs = []
                for i in range(0, len(all_decoder_inputs), decoder_batch_size):
                    batch_inputs = all_decoder_inputs[i : i + decoder_batch_size]
                    batch_outputs = decoder_model(batch_inputs, yes_no_answer=True)
                    all_decoder_outputs.extend(batch_outputs)
            # If decoder_batch_size is None, process all inputs at once
            else:
                all_decoder_outputs = decoder_model(all_decoder_inputs, yes_no_answer=True)
            # Compute rewards
            rewards = []
            idx = 0
            for count in counts:
                outputs = all_decoder_outputs[idx : idx + count]
                lbls = labels_flat[idx : idx + count]
                y_rewards = [out["answer"] == l for out, l in zip(outputs, lbls)]
                rewards.append(np.mean(y_rewards))
                idx += count
        except Exception as e:  # catch any problem with the API
            print(f"Error in decoding: {e}. Returning 0.5 rewards.")
            return [0.5] * len(prompts)

        print(f">>> avg rewards: {np.array(rewards).mean()} +/- {np.array(rewards).std()}")
        if full_print:
            # print bottlenecks by increasing reward
            for r, b in sorted(zip(rewards, completions), key=lambda x: x[0]):
                print(f'>>> reward: {r} {"".join(["*"]*int(r*10))}')
                if type(b) == list:
                    b = b[0]["content"]
                    print(b)
                else:
                    print(b)
        return rewards

    return reward_func


def create_reward_function_steer(decoder_model, decoder_batch_size=None, y_template=None, additional_bn_info=None, student_dict=None):
    """
    Create a reward function that can use student-specific information in the bottleneck.
    This version enhances the bottleneck with additional student information before decoding.
    """
    print(f"Using decoder_batch_size {decoder_batch_size}")
    if additional_bn_info:
        print(f"Using steered bottleneck with additional_bn_info: {additional_bn_info}")

    def reward_func_steer(prompts, completions, labels, y_prompts, user_id=None, **kwargs):
        """
        Compute rewards with enhanced bottlenecks that include student-specific information.
        """
        full_print = np.random.rand() < 0.05
        if full_print:
            print(">>> num_generations:", len(prompts))
            prompt_contents = [p[0]["content"] if type(p) == list else p for p in prompts]
            nb_unique_prompts = len(set(prompt_contents))
            print(">>> num_unique_prompts:", nb_unique_prompts)
            print(">>> completion:", completions[0])
            print(">>> labels:", labels[0])
            if user_id:
                print(">>> user_id:", user_id[0])

        # Normalize inputs and collect all decoder inputs
        all_decoder_inputs = []
        counts = []
        labels_flat = []

        for i, (bottleneck, y_prompt, label) in enumerate(zip(completions, y_prompts, labels)):
            # Process additional_bn_info if provided
            bottleneck_content = bottleneck[0]["content"] if isinstance(bottleneck, list) else bottleneck.copy()

            if additional_bn_info and student_dict and user_id:
                u_id = user_id[i]
                processed_bn_info = process_dynamic_info(additional_bn_info, student_dict, u_id)
                if processed_bn_info:
                    bottleneck_content = f"{bottleneck_content}\n{processed_bn_info}"
                    if full_print:
                        print(f">>> Enhanced bottleneck for user {u_id}:\n{bottleneck_content}")

            # Normalize to list
            if not isinstance(y_prompt, list):
                y_prompt = [y_prompt]
                label = [label]

            inputs = [add_instructions_to_yprompt(y, bottleneck_content, template=y_template) for y in y_prompt]
            all_decoder_inputs.extend(inputs)
            counts.append(len(inputs))
            labels_flat.extend(label)

        # Call decoder once
        try:
            # If decoder_batch_size is not None, process inputs by batches
            if decoder_batch_size is not None:
                print("Decoding in batches of size", decoder_batch_size)
                all_decoder_outputs = []
                for i in range(0, len(all_decoder_inputs), decoder_batch_size):
                    batch_inputs = all_decoder_inputs[i : i + decoder_batch_size]
                    batch_outputs = decoder_model(batch_inputs, yes_no_answer=True)
                    all_decoder_outputs.extend(batch_outputs)
            # If decoder_batch_size is None, process all inputs at once
            else:
                all_decoder_outputs = decoder_model(all_decoder_inputs, yes_no_answer=True)
            # Compute rewards
            rewards = []
            idx = 0
            for count in counts:
                outputs = all_decoder_outputs[idx : idx + count]
                lbls = labels_flat[idx : idx + count]
                y_rewards = [out["answer"] == l for out, l in zip(outputs, lbls)]
                rewards.append(np.mean(y_rewards))
                idx += count
        except Exception as e:  # catch any problem with the API
            print(f"Error in decoding: {e}. Returning 0.5 rewards.")
            return [0.5] * len(prompts)

        print(f">>> avg rewards: {np.array(rewards).mean()} +/- {np.array(rewards).std()}")
        if full_print:
            # print bottlenecks by increasing reward
            for r, b in sorted(zip(rewards, completions), key=lambda x: x[0]):
                print(f'>>> reward: {r} {"".join(["*"]*int(r*10))}')
                if type(b) == list:
                    b = b[0]["content"]
                    print(b)
                else:
                    print(b)
        return rewards

    return reward_func_steer


def format_reward_func(prompts, completions, labels, y_prompts, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<info>[\s\S]*?</info>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    # print a couple of stats about the batch rewards
    print(f"Avg rewards: {np.mean(rewards):.2f}")
    print(f"Correct formats: {sum(rewards)} / {len(rewards)}")
    if sum(rewards) > 0:
        print("Example correct format:")
        print(completion_contents[rewards.index(1)])
    if sum(rewards) < len(rewards):
        print("Example incorrect format:")
        print(completion_contents[rewards.index(0)])
    return rewards


def misconception_reward_func(prompts, completions, labels, y_prompts, **kwargs):
    """Reward function that checks if the completion mentions misconceptions in various forms."""
    # Define patterns for misconception variants (singular/plural, case variations)
    pattern = r"[Mm]isconception(?:s)?"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Check for pattern matches
    matches = [re.search(pattern, content) is not None for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]

    # Print statistics about the batch
    print(f"Misconception rewards: {np.mean(rewards):.2f}")
    print(f"Completions with misconceptions: {sum(rewards)} / {len(rewards)}")

    # Print examples of completions with and without misconceptions
    if sum(rewards) > 0:
        print("Example with misconception:")
        print(completion_contents[rewards.index(1.0)])
    if sum(rewards) < len(rewards):
        print("Example without misconception:")
        print(completion_contents[rewards.index(0.0)])

    return rewards
