import numpy as np
import tqdm
from transformers import LogitsProcessor
import torch
import pandas as pd
import os
import yaml
from datetime import datetime
import time

from src.models.lb_models.lb_base import LBModel
from src.models.lb_models.lb_gt import LBModelGT
from src.models.lb_models.lb_steer import LBModelSteer
from src.models.api_models import OpenAIModel


class YesNoLogitsProcessor(LogitsProcessor):
    def __init__(self, yes_token_id, no_token_id):
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def __call__(self, input_ids, scores):
        # Create a mask for all tokens that are not "Yes" or "No"
        mask = torch.ones_like(scores) * -float("inf")
        mask[:, [self.yes_token_id, self.no_token_id]] = 0
        # Apply the mask to the scores
        scores = scores + mask
        return scores


class ModelEvaluator:
    def __init__(self, model, test_data, batch_size=4, verbose_freq=0.2, export_bool=False, export_path="exports", **kwargs):
        self.model = model
        self.test_data = test_data
        self.batch_size = batch_size
        self.verbose_freq = verbose_freq
        self.export_bool = export_bool
        self.export_path = export_path

        # Set students_df for ground truth model if provided
        if type(self.model) is LBModelGT:  # and 'students_df' in kwargs:
            assert "students_df" in kwargs, "students_df must be provided for LBModelGT"
            self.model.set_students_df(kwargs["students_df"])

    def evaluate(self, config=None):
        correct = 0
        total = 0
        answers = []

        # Create a list to store all evaluation data for export
        export_data = []

        # Convert dictionary to list of (key, value) pairs for batching
        test_items = list(self.test_data.items())

        # Process the test data in batches
        processing_times_per_batch = []
        # for i in range(0, len(test_items), self.batch_size):
        # add tqdm
        for i in tqdm.tqdm(range(0, len(test_items), self.batch_size)):
            st = time.time()
            batch = test_items[i : i + self.batch_size]
            user_ids = [item[0] for item in batch]
            prompts = [item[1][0] for item in batch]
            true_labels = [item[1][1] for item in batch]

            # Check if we have multiple questions or just one per input
            if isinstance(self.model, (LBModel, LBModelSteer)):
                x_prompts, y_prompts = zip(*prompts)
                is_multiple_questions = (
                    any(isinstance(getattr(y_prompt, "list", lambda: y_prompt)(), list) for y_prompt in y_prompts)
                    if hasattr(y_prompts, "__iter__")
                    else False
                )
            else:
                is_multiple_questions = isinstance(
                    prompts[0], list
                )

            is_lbm = isinstance(self.model, (LBModel, LBModelGT, LBModelSteer))

            # Get predictions
            if type(self.model) is LBModel:
                x_prompts, y_prompts = zip(*prompts)
                outputs = self.model(input_text=list(x_prompts), new_question_text=list(y_prompts))
            elif type(self.model) is LBModelGT:
                x_prompts, y_prompts = zip(*prompts)
                outputs = self.model(new_question_text=list(y_prompts), user_ids=user_ids)
            elif type(self.model) is LBModelSteer:
                x_prompts, y_prompts = zip(*prompts)
                outputs = self.model(input_text=list(x_prompts), new_question_text=list(y_prompts), user_id=list(user_ids))
            else:
                if is_multiple_questions:
                    outputs = [self.model(prompt) for prompt in prompts]
                else:
                    outputs = self.model(list(prompts))

            # Ensure outputs is a list of dictionaries
            if not isinstance(outputs, list):
                outputs = [outputs]

            if is_multiple_questions and is_lbm:
                predictions = [[d["answer"] for d in output["decoder_outputs"]] for output in outputs]
                logits = [[d.get("logits", None) for d in output["decoder_outputs"]] for output in outputs]
            elif is_multiple_questions:
                # # If multiple questions, we need to extract the answers from the outputs
                predictions = [[o["answer"] for o in output] for output in outputs]
                logits = [[o.get("logits", None) for o in output] for output in outputs]
            else:
                predictions = [o["answer"] for o in outputs]
                logits = [o.get("logits", None) for o in outputs]

            # Compare predictions with true labels
            for idx, (user_id, true_label, prediction, logit, output, prompt) in enumerate(
                zip(user_ids, true_labels, predictions, logits, outputs, prompts)
            ):
                if is_multiple_questions:
                    avg_correct = np.mean([p == l for p, l in zip(prediction, true_label)])
                    correct += avg_correct
                else:
                    correct += int(prediction.strip() == true_label)
                total += 1
                answers.append((user_id, true_label, prediction))

                # Collect data for export
                if self.export_bool:
                    export_item = {
                        "user_id": user_id,
                        "prompt": prompt if isinstance(prompt, str) else str(prompt),
                        "true_label": true_label if isinstance(true_label, str) else str(true_label),
                        "prediction": prediction if isinstance(prediction, str) else str(prediction),
                        "logit": logit if isinstance(logit, str) else str(logit),
                        "correct": avg_correct if is_multiple_questions else int(prediction.strip() == true_label),
                    }

                    # Add all output fields to the export data
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if key != "answer":  # Already captured in prediction
                                export_item[f"output_{key}"] = str(value)
                    elif isinstance(output, list):
                        export_item[f"output_full"] = str(output)

                    export_data.append(export_item)

            if np.random.rand() < self.verbose_freq:
                print("Full output:")
                if isinstance(outputs[0], dict):
                    for k, v in outputs[0].items():
                        print(f">>>{k}: {v}\n")
                elif isinstance(outputs[0], list):
                    for k, output in enumerate(outputs[0]):
                        print(f"Output {k}: {output}")
                else:
                    print(f"Output: {outputs[0]}")
            if (type(self.model) is OpenAIModel) or (type(self.model) is LBModel):
                # print total price every 20 iterations
                if i % 20 == 0:
                    print(f"Total price: {self.model.total_price}")

            processing_times_per_batch.append(time.time() - st)
            print(
                f"Batch {i} processed in {processing_times_per_batch[-1]:.2f} seconds (s/sample: {processing_times_per_batch[-1] / self.batch_size:.2f})"
            )

        # Export data if requested
        if self.export_bool and export_data:
            export_dir = self._export_data(export_data, config)
        else:
            export_dir = None

        return answers, correct / total, export_dir

    def _export_data(self, export_data, config):
        """Export evaluation data to CSV and config to YAML."""
        # Create timestamp for folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(self.export_path, timestamp)

        # Create directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)

        # Export data to CSV
        df = pd.DataFrame(export_data)
        csv_path = os.path.join(export_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported evaluation data to {csv_path}")

        # Export config to YAML if provided
        if config:
            yaml_path = os.path.join(export_dir, "config.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(config, f)
            print(f"Exported configuration to {yaml_path}")
        return export_dir
