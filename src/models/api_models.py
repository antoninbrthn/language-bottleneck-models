"""Load and run models via the Azure OpenAI API."""

from typing import Tuple, Union
from openai import OpenAI, AzureOpenAI
import os
import asyncio
import aiohttp

from src.utils.llm_pricing import get_price

# Maps public model names to Azure deployment names.
# Update these to match your Azure OpenAI deployment names.
MODEL_TO_NAME = {
    "gpt-4o": os.getenv("AZURE_GPT4O_DEPLOYMENT", "gpt-4o"),
    "gpt-4o-mini": os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini"),
    "gpt-5": os.getenv("AZURE_GPT5_DEPLOYMENT", "gpt-5"),
}


API_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5"]


def load_api_model(model_config):
    model_name = MODEL_TO_NAME[model_config.name]
    model = OpenAIModel(model_name=model_name, **model_config)
    return model


class OpenAIModel:
    def __init__(self, model_name, max_cot_tokens=1024, use_cot=False, use_batch=True, **kwargs):
        """
        Initializes the model with an AzureOpenAIClient object.

        Args:
            model_name (str): The name of the model.
            max_cot_tokens (int): Maximum tokens for chain of thought reasoning.
            use_cot (bool): Flag to indicate whether to use chain of thought reasoning.
        """
        self.model_name = model_name
        self.max_cot_tokens = max_cot_tokens
        self.use_cot = use_cot
        self.use_batch = use_batch
        self.client = AzureOpenAIClient(self.model_name)
        self.total_price = 0
        self.only_certain_answers = kwargs.get("only_certain_answers", False)

    def __call__(self, input_text: Union[str, list], yes_no_answer=False):
        """
        Handles user input, sends it to the Azure OpenAI API, and returns the assistant's response.

        Args:
            input_text (str): The user's input message.
            yes_no_answer (bool): Flag to indicate whether the answer should be "Yes" or "No". If so will clean up answer before returning it.

        Returns:
            dict: A dictionary containing the assistant's response and reasoning if use_cot is enabled.
        """
        if self.use_cot:
            answers, reasonings = self._cot_call(input_text)
            assert len(answers) == len(reasonings), "Answers and reasonings should have the same length"
            if yes_no_answer:
                answers = [self._clean_answer(ans) for ans in answers]
            return [{"answer": ans, "reasoning": res} for ans, res in zip(answers, reasonings)]
        else:
            print("In call", yes_no_answer)
            if self.only_certain_answers:
                print("WARNING: Prompting on certain answers")
                input_text = [
                    f"{text}\nOne last thing: please only answer by Yes/No **if you can be absolutely sure given the information provided**. If there is any randomness or doubt involved, reply with 'null' instead of 'Yes' or 'No'."
                    for text in input_text
                ]
            answers = self._generate(input_text=input_text)  # directly generate response
            if yes_no_answer:
                answers = [self._clean_answer(ans) for ans in answers]
            return [{"answer": ans} for ans in answers]

    def _cot_call(self, input_text: Union[str, list]) -> Tuple[list, list]:
        """Generate a response using chain of thought reasoning.
        Can take either a string (single prompt) or a list of strings (multiple prompts) as input,
        but will always return a list of responses.
        """
        if type(input_text) == str:
            input_text = [input_text]
        cot_prompts = [txt + "\nThink step by step and lay out your thinking process carefully." for txt in input_text]
        messages = [[{"role": "user", "content": cot_prompt}] for cot_prompt in cot_prompts]
        reasonings = self._generate(messages=messages)

        if type(reasonings) == str:  # might be a single string if one sample is provided
            reasonings = [reasonings]
        # Final prompt for the answer
        for i, reasoning in enumerate(reasonings):
            messages[i].append({"role": "assistant", "content": reasoning})
            messages[i].append(
                {
                    "role": "user",
                    "content": "Now please give your final answer by writing 'Yes' or 'No' and nothing else.",
                }
            )
        answers = self._generate(messages=messages)
        if type(answers) == str:  # might be a single string if one sample is provided
            answers = [answers]
        if (len(answers) == 1) and (len(reasonings) == 1):
            return answers[0], reasonings[0]
        return answers, reasonings  # Return both answer and reasoning

    def _generate(self, *, input_text: Union[str, list] = None, messages=None) -> list:
        """Generate a response using the model.
        Can take either a string (single prompt) or a list of strings (multiple prompts) as input,
        but will always return a list of responses.
        """
        if type(input_text) == str:  # single sample
            input_text = [input_text]
        assert messages is not None or input_text is not None, "Either messages or input_text must be provided for batch processing"

        # Process input_text into messages format if provided
        if messages is None:
            messages = [[{"role": "user", "content": text}] for text in input_text]

        if self.use_batch:  # batch processing
            print(f"Using batch to generate {len(messages)} inputs")
            try:
                outputs = asyncio.run(self._batch_generate(messages_list=messages))
            except KeyError as e:
                print("WARNING: Probably exceeded OpenAI API rate limit. Retrying in 60 seconds..")
                print("error", e)
                import time

                time.sleep(60)
                return self._generate(input_text=input_text, messages=messages)
            return outputs
        else:  # process one by one
            output = []
            for message in messages:
                completion = self.client.get_completion(messages=message, max_tokens=self.max_cot_tokens)
                self.update_price(completion)
                output.append(completion.choices[0].message.content)
            return output[0] if len(output) == 1 else output

    def update_price(self, completion):
        price = get_price(completion)
        self.total_price += price
        if self.total_price > 10:
            print(f"Total price: {self.total_price}")
        if self.total_price > 100:
            print("Price exceeds 100 dollars. Aborting.")

    def _clean_answer(self, answer: str) -> str:
        """Clean up and validate the answer.

        Args:
            answer (str): The raw answer from the model

        Returns:
            str: Cleaned answer ("Yes" or "No")

        Raises:
            Warning: If the answer cannot be cleaned to "Yes" or "No"
        """
        # Get last line, remove any punctuation and whitespace
        # cleaned = answer.split('/n')[-1].strip().rstrip('.,!?').strip()
        # Remove any punctuation and whitespace
        cleaned = answer.strip().rstrip(".,!?").strip()

        # Convert to title case for consistency
        cleaned = cleaned.title()

        if cleaned not in ["Yes", "No"]:
            import warnings

            print(f"Unexpected answer format: '{answer}'. Expected 'Yes' or 'No'.")
            # Default to non-cleaned answer
            return answer
        if cleaned != answer:
            print(f"Answer cleaned from '{answer}' to '{cleaned}'")
        return cleaned

    # async calls
    async def _generate_async(self, input_text=None, messages=None):
        """Async version of _generate using AzureOpenAIClient."""
        assert messages is not None or input_text is not None, "Either messages or input_text must be provided for batch processing"
        assert (input_text is None) != (messages is None), "Provide either messages_list or input_list, not both"

        if messages is None:
            messages = [{"role": "user", "content": input_text}]

        try:
            # max_completion_tokens for gpt-5, otherwise max_tokens
            args = {"max_completion_tokens": self.max_cot_tokens} if "gpt-5" in self.model_name else {"max_tokens": self.max_cot_tokens}
            if ("gpt-5" in self.model_name) and (not self.use_cot):
                args["reasoning_effort"] = "minimal"
            completion = await self.client.get_completion_async(messages=messages, args=args)
            return completion
        except Exception as e:
            print("Error during API call:", e)
            return f"Error: {e}"

    async def _batch_generate(self, input_list=None, messages_list=None):
        """Executes multiple requests in parallel using asyncio."""
        assert messages_list is not None or input_list is not None, "Either messages_list or input_list must be provided for batch processing"
        assert (messages_list is None) != (input_list is None), "Provide either messages_list or input_list, not both"

        if input_list is not None:
            tasks = [self._generate_async(input_text=text) for text in input_list]
        else:
            tasks = [self._generate_async(messages=messages) for messages in messages_list]
        results = await asyncio.gather(*tasks)
        output = []
        for completion in results:
            self.update_price(completion)
            output.append(completion["choices"][0]["message"]["content"])
        return output[0] if len(output) == 1 else output


class AzureOpenAIClient:
    def __init__(self, model="gpt-4o-mini-glob1"):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
        )
        self.model = model

    def get_completion(self, messages, max_tokens=1000, args={}, **kwargs):
        """Synchronous API call for backward compatibility."""
        # if 'gpt-5' not in self.model:
        if "max_completion_tokens" not in kwargs.keys() and "max_completion_tokens" not in args.keys():
            args["max_tokens"] = max_tokens
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=123,
            **args,
            **kwargs,
        )

    async def get_completion_async(self, messages, args={"max_tokens": 1000}):
        """Asynchronous API call to support parallel processing."""
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01"
        headers = {
            "Authorization": f"Bearer {os.getenv('AZURE_OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages, **args}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                result = await response.json()
                return result
