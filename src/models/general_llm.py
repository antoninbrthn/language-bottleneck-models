import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src.models.api_models import MODEL_TO_NAME, AzureOpenAIClient
from src.utils.llm_pricing import get_price


class LLMText:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", system_prompt=None):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._is_gpt_model = ("gpt" in self.model_id) and self.model_id != "gpt2"
        self._system_prompt = system_prompt
        self.total_price = 0.0

        # use alias if necessary
        if self._is_gpt_model and model_id in MODEL_TO_NAME:
            self.model_id = MODEL_TO_NAME[model_id]
            print("Using model alias:", self.model_id)
        if self._is_gpt_model:
            self._client = AzureOpenAIClient(self.model_id)
        else:
            # Load the model and processor
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

    def generate_response(self, text="", max_new_tokens=30, args={}):
        # Create the chat template
        messages = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})

        if self._is_gpt_model:
            args.update({"max_tokens": max_new_tokens})
            completion = self._client.get_completion(messages=messages, **args)
            self.total_price += get_price(completion)
            response = completion.choices[0].message.content
            return response
        else:
            # Prepare the input text
            input_text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            processor_inputs = {"text": input_text, "add_special_tokens": False, "return_tensors": "pt"}
            inputs = self._processor(**processor_inputs).to(self._model.device)

            # Generate the response
            output = self._model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
            prompt_length = inputs["input_ids"].shape[1]
            return self._processor.decode(output[0][prompt_length:], skip_special_tokens=True)

    def generate_response_batch(self, texts="", max_new_tokens=30, args={}):
        """
        Repeat function above but for a batch of inputs. Build a list of inputs and call the model once.
        """
        assert not self._is_gpt_model, "Batch response generation is only supported for non-GPT models."
        # Prepare the input texts
        messages_list = []
        for text in texts:
            messages = []
            if self._system_prompt is not None:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": text})
            messages_list.append(messages)

        input_texts = [self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]

        processor_inputs = self._processor(
            text=input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # Generate the responses
        output = self._model.generate(processor_inputs["input_ids"], attention_mask=processor_inputs["attention_mask"], max_new_tokens=max_new_tokens)
        responses = []
        for i in range(len(texts)):
            prompt_length = (processor_inputs["attention_mask"][i] == 1).sum().item()
            response = self._processor.decode(output[i][prompt_length:], skip_special_tokens=True)
            responses.append(response)
        return responses

    def generate_structured_response(self, text="", max_new_tokens=30, args={}):
        """
        # From openai docs:
        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        completion = client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
                {"role": "user", "content": "how can I solve 8x + 7 = -23"}
            ],
            response_format=MathReasoning,
        )

        """
        assert self._is_gpt_model, "Structured response generation is only supported for GPT models."
        messages = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})

        args.update({"response_format": args.get("response_format", LLMMisconceptionJudgeResponse)})
        completion = self._client.client.chat.completions.parse(
            model=self._client.model,
            messages=messages,
            max_tokens=max_new_tokens,
            seed=123,
            **args,
        )
        self.total_price += self._client.get_price(completion)
        response = completion.choices[0].message.parsed
        return response


from pydantic import BaseModel


class MisconceptionJudgeResponse(BaseModel):
    misc_id: int
    score: int


class LLMMisconceptionJudgeResponse(BaseModel):
    output: list[MisconceptionJudgeResponse]
