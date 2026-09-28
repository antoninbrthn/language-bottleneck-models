from typing import List, Dict
from src.prompts.prompt_generator import format_question, format_construct

def split_into_batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def batch_to_history_text(batch: List[Dict], contains) -> str:
    lines = ["The student answered the following questions:\n"]
    for entry in batch:
        correctness = "correctly" if entry["correct"] else "incorrectly"
        q_txt = format_question(entry, contains)
        c_txt = format_construct(entry, contains)
        if q_txt:
            lines.append(f"Question {q_txt}, with construct {c_txt}: answered {correctness}.\n")
        else:
            lines.append(f"Question with construct {c_txt}: answered {correctness}.\n")
    return "".join(lines)


def load_and_prepare_data_iterative(loader, prompt_gen, n_y_prompts=1):
    data = loader.load_data()
    trajectories, user_ids = loader.preprocess(data)
    train_traj, test_traj = loader.split_data(trajectories)
    train_ids,  test_ids  = loader.split_data(user_ids)

    train_dict, test_dict = {}, {}
    if n_y_prompts > 1:
        for uid, traj in zip(train_ids, train_traj):
            x_traj, y_entries = traj[:-n_y_prompts], traj[-n_y_prompts:]
            x_prompt = list(x_traj)                      # keep *list of dicts*
            y_prompt = [prompt_gen.create_prompt_bottleneck([], e)[1]
                        for e in y_entries]
            labels   = ["Yes" if e["correct"] else "No" for e in y_entries]
            train_dict[uid] = ((x_prompt, y_prompt), labels)
        # repeat for test_dict …
        for uid, traj in zip(test_ids, test_traj):
            x_traj, y_entries = traj[:-n_y_prompts], traj[-n_y_prompts:]
            x_prompt = list(x_traj)                      # keep *list of dicts*
            y_prompt = [prompt_gen.create_prompt_bottleneck([], e)[1]
                        for e in y_entries]
            labels   = ["Yes" if e["correct"] else "No" for e in y_entries]
            test_dict[uid] = ((x_prompt, y_prompt), labels)
    else:
        raise NotImplementedError("n_y_prompts=1 case not implemented in iterative-type loading")
        for uid, traj in zip(train_ids, train_traj):
            x_prompt, y_prompt = prompt_gen.create_prompt_bottleneck([], traj[-1])
            train_dict[uid] = (list(traj[:-1]), y_prompt)   # store trajectory list
        # repeat for test_dict …
        for uid, traj in zip(test_ids, test_traj):
            x_prompt, y_prompt = prompt_gen.create_prompt_bottleneck([], traj[-1])
            test_dict[uid] = (list(traj[:-1]), y_prompt)

    return train_dict, test_dict