"""Load and run models locally via HuggingFace (e.g. Llama, Gemma, Qwen)."""

from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
import os
import torch
from torch.nn.utils.rnn import pad_sequence

# Shorthand aliases for commonly used HuggingFace model IDs.
NAME_TO_HF = {
    "llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2-2B": "google/gemma-2-2b-it",
    "ministral-8B": "mistralai/Ministral-8B-Instruct-2410",
}
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_hf_model(model_config):
    model_config_dict = OmegaConf.to_object(model_config)
    name = model_config_dict.pop("name")
    hf_name = NAME_TO_HF[name] if name in NAME_TO_HF.keys() else name
    use_chat = model_config_dict.get("use_chat", True)
    if use_chat:
        # model = HFModelChat(model=hf_name, checkpoint=checkpoint, peft_checkpoint=peft_checkpoint, use_cot=use_cot, dtype=dtype, max_cot_tokens=max_cot_tokens, **model_config)
        model = HFModelChat(model=hf_name, **model_config_dict)
    else:
        # model = HFModel(model=hf_name, checkpoint=checkpoint, peft_checkpoint=peft_checkpoint, use_cot=use_cot, dtype=dtype, max_cot_tokens=max_cot_tokens, **model_config)
        model = HFModel(model=hf_name, **model_config_dict)
    # elif "pipe" in name:
    # model = HFModelChatPipe(model_name=hf_name)
    return model


class HFModel:
    def __init__(self, model, checkpoint=None, tokenizer=None, on_cpu=False, use_cot=False, dtype=torch.bfloat16, max_cot_tokens=1, **kwargs):
        """
        Initialize with either a model name or a pre-loaded model and tokenizer
        """
        self.dtype = eval(dtype) if type(dtype) == str else dtype
        if checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
            print("Loaded model from checkpoint:", checkpoint)
        elif isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model, token=HF_TOKEN, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                token=HF_TOKEN,
                torch_dtype=self.dtype,
                #   device_map="auto",
                low_cpu_mem_usage=True,  # required for deepseed
            )
        else:
            # Assume pre-loaded model and tokenizer
            self.model = model
            self.tokenizer = tokenizer

        self.use_cot = use_cot
        self.max_cot_tokens = max_cot_tokens

        # Set device
        if torch.cuda.is_available() and not on_cpu:
            print("Using GPU")
            self.model.to("cuda")
        else:
            print("WARNING: using CPU")
            self.model.to("cpu")

    def _generate(self, input_text, max_new_tokens):
        """Helper method to generate text"""
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prompt_length = input_ids.shape[-1]
        generated_tokens = output[0][prompt_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _generate_batch(self, input_texts, max_new_tokens):
        """Helper method to generate text for a batch of inputs"""
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        results = []
        for i, output in enumerate(outputs):
            prompt_length = input_ids[i].shape[0]
            generated_tokens = output[prompt_length:]
            results.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))

        return results

    def _direct_call(self, input_text):
        """Generate a direct Yes/No response"""
        return self._generate(input_text, max_new_tokens=1)

    def _cot_call(self, input_text):
        """Generate a response using chain of thought reasoning"""
        # First completion - get the reasoning
        cot_prompt = input_text + "\nThink step by step and lay out your thinking process carefully. Then, give your final answer as 'Yes' or 'No'"
        reasoning = self._generate(cot_prompt, max_new_tokens=self.max_cot_tokens)

        # Second completion - get the final Yes/No
        final_prompt = reasoning + "\nNow give your final answer as 'Yes' or 'No'. Final Answer: "
        return self._generate(final_prompt, max_new_tokens=1)

    def _direct_call_batch(self, input_texts):
        """Generate direct Yes/No responses for a batch of inputs"""
        return self._generate_batch(input_texts, max_new_tokens=1)

    def _cot_call_batch(self, input_texts):
        """Generate responses using chain of thought reasoning for a batch of inputs"""
        # First completion - get the reasoning for each input
        cot_prompts = [
            text + "\nThink step by step and lay out your thinking process carefully. Then, give your final answer as 'Yes' or 'No'"
            for text in input_texts
        ]
        reasonings = self._generate_batch(cot_prompts, max_new_tokens=self.max_cot_tokens)

        # Second completion - get the final Yes/No for each input
        final_prompts = [reasoning + "\nNow give your final answer as 'Yes' or 'No'. Final Answer: " for reasoning in reasonings]
        answers = self._generate_batch(final_prompts, max_new_tokens=1)

        return answers, reasonings

    def _clean_answer(self, answer: str) -> str:
        """Clean up and validate the answer.

        Args:
            answer (str): The raw answer from the model

        Returns:
            str: Cleaned answer ("Yes" or "No")

        Raises:
            Warning: If the answer cannot be cleaned to "Yes" or "No"
        """
        # Remove any punctuation and whitespace
        cleaned = answer.strip().rstrip(".,!?").strip()

        # Convert to title case for consistency
        cleaned = cleaned.title()

        if cleaned not in ["Yes", "No"]:
            import warnings

            warnings.warn(f"Unexpected answer format: '{answer}'. Expected 'Yes' or 'No'.")
            # Default to non-cleaned answer
            return answer

        return cleaned

    def __call__(self, input_text, yes_no_answer=False):
        """Generate response based on use_cot setting

        Args:
            input_text: Single text input or list of text inputs
            yes_no_answer: Whether to clean up answer to be "Yes" or "No"
        """
        # Handle batch input
        if isinstance(input_text, list):
            if self.use_cot:
                answers, reasonings = self._cot_call_batch(input_text)
                if yes_no_answer:
                    answers = [self._clean_answer(ans) for ans in answers]
                return answers
            else:
                answers = self._direct_call_batch(input_text)
                if yes_no_answer:
                    answers = [self._clean_answer(ans) for ans in answers]
                return answers

        # Handle single input (original behavior)
        else:
            if self.use_cot:
                answer = self._cot_call(input_text)
                if yes_no_answer:
                    answer = self._clean_answer(answer)
                return answer
            answer = self._direct_call(input_text)
            if yes_no_answer:
                answer = self._clean_answer(answer)
            return answer


class HFModelChat:
    """Use chat interface via tokenizer.apply_chat_template with constrained Yes/No output"""

    def __init__(
        self,
        model,
        checkpoint=None,
        peft_checkpoint=None,
        tokenizer=None,
        on_cpu=False,
        use_cot=False,
        dtype=torch.bfloat16,
        max_cot_tokens=256,
        set_cuda_device=None,
        attn_implementation=None,
        pad_to_multiple_of=None,
        **kwargs,
    ):
        """Initialize with either a model name or pre-loaded model and tokenizer"""
        self.dtype = eval(dtype) if type(dtype) == str else dtype
        self.pad_to_multiple_of = pad_to_multiple_of  # to be passed to tokenizer calls, fixes bug with Gemma3 models
        self.attn_implementation = attn_implementation
        if checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
            print("Loaded model from checkpoint:", checkpoint)
        if peft_checkpoint:
            from peft import PeftModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model, token=HF_TOKEN, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                token=HF_TOKEN,
                torch_dtype=self.dtype,
                #   device_map="auto",
                low_cpu_mem_usage=True,  # required for deepseed
            )
            self.model = PeftModelForCausalLM.from_pretrained(self.model, peft_checkpoint)
            print("Loaded model from PEFT checkpoint:", peft_checkpoint)
        elif isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                token=HF_TOKEN,
                torch_dtype=self.dtype,
                #   device_map="auto",
                low_cpu_mem_usage=True,  # required for deepseed
                attn_implementation=self.attn_implementation,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model, token=HF_TOKEN, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.use_cot = use_cot
        self.max_cot_tokens = max_cot_tokens

        # Get token IDs for "Yes" and "No"
        self.yes_token_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
        self.no_token_ids = self.tokenizer.encode(" No", add_special_tokens=False)

        # For single token comparison (if tokenizer splits these into multiple tokens)
        self.yes_token_id = self.yes_token_ids[0] if len(self.yes_token_ids) > 0 else None
        self.no_token_id = self.no_token_ids[0] if len(self.no_token_ids) > 0 else None

        # Set device
        if torch.cuda.is_available() and not on_cpu:
            device = f"cuda:{set_cuda_device}" if set_cuda_device is not None else "cuda:0"
            self.model.to(device)
        else:
            print("WARNING: using CPU")
            self.model.to("cpu")

    def _generate(self, input_text=None, messages=None, max_new_tokens=512, force_yes_no=False):
        """Helper method to generate text from messages with optional Yes/No constraint

        Args:
            input_text: Single text input or list of text inputs
            messages: Single message list or list of message lists
            max_new_tokens: Maximum number of tokens to generate
            force_yes_no: Whether to force Yes/No output
            batch: Whether inputs are provided as a batch
        """
        # Handle batch processing
        # if batch:
        if type(input_text) == str:  # single sample
            input_text = [input_text]
        assert messages is not None or input_text is not None, "Either messages or input_text must be provided for batch processing"

        # Process input_text into messages format if provided
        if messages is None:
            messages = [[{"role": "user", "content": text}] for text in input_text]

        # Process each message list and tokenize
        all_input_ids = []
        all_attention_masks = []
        for msg in messages:
            inputs = self.tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenizer_kwargs={"pad_to_multiple_of": self.pad_to_multiple_of, "padding_side": "left"} if self.pad_to_multiple_of else {},
            )
            all_input_ids.append(inputs["input_ids"].squeeze(0))
            all_attention_masks.append(inputs["attention_mask"].squeeze(0))

        # Pad sequences to the same length
        batch_input_ids = pad_sequence_left(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        batch_attention_mask = pad_sequence_left(all_attention_masks, batch_first=True, padding_value=0).to(self.model.device)

        if not force_yes_no:
            # Standard batch generation
            outputs = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Process each output separately
            results = []
            for i, output in enumerate(outputs):
                prompt_length = batch_input_ids[i].shape[0]
                generated_tokens = output[prompt_length:]
                results.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))
            return results

        else:
            # Get logits for next token only for each input in batch
            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits[:, -1, :]  # Get logits for the next token for each item in batch

                # Extract logits for Yes/No tokens for each item in batch
                results = []
                for i in range(logits.shape[0]):
                    yes_logit = logits[i, self.yes_token_id].item() if self.yes_token_id is not None else float("-inf")
                    no_logit = logits[i, self.no_token_id].item() if self.no_token_id is not None else float("-inf")

                    # Determine the answer and return both the answer and logits
                    answer = "Yes" if yes_logit > no_logit else "No"
                    results.append((answer, {"Yes": yes_logit, "No": no_logit}))
                return results

    def _direct_call_batch(self, input_texts):
        """Generate direct Yes/No responses with logits for a batch of inputs"""
        messages = [[{"role": "user", "content": text}] for text in input_texts]
        results = self._generate(messages=messages, max_new_tokens=1, force_yes_no=True)
        return results  # List of (answer, logits) tuples

    def _cot_call_batch(self, input_texts):
        """Generate responses using chain of thought reasoning for a batch of inputs"""
        # First completion - get the reasoning for each input
        cot_prompts = [
            text + "\nThink step by step and lay out your thinking process carefully but succintly. Then, give your final answer as 'Yes' or 'No'."
            for text in input_texts
        ]
        messages = [[{"role": "user", "content": prompt}] for prompt in cot_prompts]
        reasonings = self._generate(messages=messages, max_new_tokens=self.max_cot_tokens, force_yes_no=False)

        # Second completion - get the final Yes/No for each input
        final_messages = []
        for i, reasoning in enumerate(reasonings):
            final_messages.append(
                [
                    {"role": "user", "content": cot_prompts[i]},
                    {"role": "assistant", "content": reasoning},
                    {"role": "user", "content": "Now give your final answer by writing 'Yes' or 'No' and nothing else."},
                ]
            )

        results = self._generate(messages=final_messages, max_new_tokens=1, force_yes_no=True)

        # Combine results with reasoning
        combined_results = []
        for (answer, logits), reasoning in zip(results, reasonings):
            combined_results.append((answer, logits, reasoning))

        return combined_results

    def _clean_answer(self, answer: str) -> str:
        """Clean up and validate the answer.

        Args:
            answer (str): The raw answer from the model

        Returns:
            str: Cleaned answer ("Yes" or "No")

        Raises:
            Warning: If the answer cannot be cleaned to "Yes" or "No"
        """
        # Remove any punctuation and whitespace
        cleaned = answer.strip().rstrip(".,!?").strip()

        # Convert to title case for consistency
        cleaned = cleaned.title()

        if cleaned not in ["Yes", "No"]:
            import warnings

            warnings.warn(f"Unexpected answer format: '{answer}'. Expected 'Yes' or 'No'.")
            # Default to non-cleaned answer
            return answer

        return cleaned

    def __call__(self, input_text, yes_no_answer=False):
        """Generate response based on use_cot setting

        Args:
            input_text: Single text input or list of text inputs
            yes_no_answer: Whether to clean up answer to be "Yes" or "No"
        """
        # Handle batch input
        if type(input_text) == str:  # single sample
            input_text = [input_text]

        if self.use_cot:
            results = self._cot_call_batch(input_text)
            if yes_no_answer:  # Yes no is already forced in _generate but let's clean it again for consistency
                return [{"answer": self._clean_answer(answer), "logits": logits, "reasoning": reasoning} for answer, logits, reasoning in results]
            return [{"answer": answer, "logits": logits, "reasoning": reasoning} for answer, logits, reasoning in results]
        else:
            results = self._direct_call_batch(input_text)
            if yes_no_answer:  # Yes no is already forced in _generate but let's clean it again for consistency
                return [{"answer": self._clean_answer(answer), "logits": logits} for answer, logits in results]
            return [{"answer": answer, "logits": logits} for answer, logits in results]


def pad_sequence_left(sequences, batch_first=True, padding_value=0):
    # Reverse each sequence
    flipped = [torch.flip(seq, dims=[0]) for seq in sequences]

    # Apply pad_sequence
    padded = pad_sequence(flipped, batch_first=batch_first, padding_value=padding_value)

    # Flip back to restore original order
    return torch.flip(padded, dims=[1]) if batch_first else torch.flip(padded, dims=[0])
