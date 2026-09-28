import re
from src.models.lb_models.utils import add_instructions_to_xprompt
from src.prompts.prompt_utils import load_prompt_template
from src.models.api_models import OpenAIModel
from src.models.model_loader import load_any_model
from src.models.lb_models.utils import add_instructions_to_yprompt


########## Base model
class LBModel:
    """
    A "Language Bottleneck" model that uses:
      1) An encoder model to summarize or transform the input into a free-form
         textual bottleneck enclosed in <info>...</info>.
      2) A decoder model that takes the bottleneck (and a short instruction)
         to produce the final output in the form of a dictionary.

    Typical usage:
        encoder_config = {"type": "hf", "name": "llama3.2-3B", "use_cot": False, ...}
        decoder_config = {"type": "api", "name": "gpt-4o", "use_cot": True, ...}
        
        lbm = LBModel(encoder_config, decoder_config,
                      max_bottleneck_tokens=128,
                      intermediate_prompt="Predict the answer based on the info below.")
        output = lbm("My original input here")
        # output -> { "answer": ..., "reasoning": ..., "logits": ..., "bottleneck": ... }
    """

    def __init__(
        self,
        encoder_config,
        decoder_config,
        max_bottleneck_tokens=256,
        x_template_file="base_x.yaml",
        y_template_file="base_y.yaml",
        **kwargs,
    ):
        """
        :param encoder_config: Configuration dictionary for the encoder (summarizer) model.
        :param decoder_config: Configuration dictionary for the decoder (final prediction) model.
        :param max_bottleneck_tokens: Suggested maximum size of the encoder's <info> summary.
        :param x_template_file: Filename for the encoder prompt template
        :param y_template_file: Filename for the decoder prompt template
        """
        self.encoder = load_any_model(encoder_config)
        self.decoder = load_any_model(decoder_config)
        self.max_bottleneck_tokens = max_bottleneck_tokens
        
        # Load templates
        self.x_template = load_prompt_template(filename=x_template_file)
        self.y_template = load_prompt_template(filename=y_template_file)

    def _extract_info_section(self, text):
        """
        Extract the substring enclosed by <info>...</info> from text.
        If none found, returns an empty string.
        """
        match = re.search(r"<info>(.*?)</info>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        print('WARNING: could not find <info> section in encoder output', text)
        return ""

    def __call__(self, input_text, new_question_text):
        """
        Run the LBModel on the given input text:
          1) Use self.encoder to produce a textual summary in <info>...</info>.
          2) Extract the summary (bottleneck).
          3) Use self.decoder with (intermediate_prompt + bottleneck).
          4) Return the decoder's output plus the 'bottleneck' for debugging.
        
        :param input_text: str or List[str]
        :param new_question_text: str or List[str]
        :return: A dictionary typically including 'answer' and optional 'reasoning', 'logits', etc.
        """
        if type(input_text) == list:
            assert type(new_question_text) == list
            if type(new_question_text[0]) == list:
                return self.__call_batch_multiple_y__(input_text, new_question_text)
            else:
                return self.__call_batch__(input_text, new_question_text)
        else:
            if type(new_question_text) == list:
                return self.__call_batch_multiple_y__([input_text], [new_question_text])
            
        # 1) Produce a bottleneck summary from the encoder.
        # if cot, prompt to think step by step and lay out thinking process before final <info> summary
        encoder_prompt = add_instructions_to_xprompt(input_text, self.encoder.use_cot, 
                                                    max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                    template=self.x_template)
        # encoder_output = self.encoder(encoder_prompt)
        args = {'max_new_tokens': self.encoder.max_cot_tokens} if type(self.encoder) != OpenAIModel else {}
        raw_encoder = self.encoder._generate(input_text=encoder_prompt, **args)
        
        # 2) Extract the <info> section from the encoder's output:
        bottleneck = self._extract_info_section(raw_encoder)

        # 3) Construct a decoding prompt:
        #    The user can vary how the second model is prompted. 
        #    For example, they might want a short instruction, or custom structure, etc.
        decoder_prompt = add_instructions_to_yprompt(new_question_text, bottleneck, 
                                                    template=self.y_template)
        
        # 4) Call the decoder with the combined prompt to get the final output:
        decoder_output = self.decoder(decoder_prompt, yes_no_answer=True)

        # 5) Merge the final output with the extracted bottleneck (for inspection or debugging):
        final_output = dict(decoder_output)  # make a copy
        final_output["raw_encoder"] = raw_encoder
        final_output["bottleneck"] = bottleneck

        return final_output
    
    def __call_batch__(self, input_texts, new_question_texts):
        """
        Run the LBModel on the given input text:
          1) Use self.encoder to produce a textual summary in <info>...</info>.
          2) Extract the summary (bottleneck).
          3) Use self.decoder with (intermediate_prompt + bottleneck).
          4) Return the decoder's output plus the 'bottleneck' for debugging.
        
        :param input_texts: List[str]
        :param new_question_texts: List[str]
        :return: A dictionary typically including 'answer' and optional 'reasoning', 'logits', etc.
        """
        # 1) Produce a bottleneck summary from the encoder.
        # if cot, prompt to think step by step and lay out thinking process before final <info> summary
        encoder_prompts = [add_instructions_to_xprompt(input_text, self.encoder.use_cot, 
                                                     max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                     template=self.x_template) 
                          for input_text in input_texts]
        # encoder_output = self.encoder(encoder_prompt)
        args = {'max_new_tokens': self.max_bottleneck_tokens} if type(self.encoder) != OpenAIModel else {}

        raw_encoder_outputs = self.encoder._generate(input_text=encoder_prompts, **args)

        
        # 2) Extract the <info> section from the encoder's output:
        bottlenecks = [self._extract_info_section(raw_encoder_output) for raw_encoder_output in raw_encoder_outputs]

        # 3) Construct a decoding prompt:
        #    The user can vary how the second model is prompted. 
        #    For example, they might want a short instruction, or custom structure, etc.
        decoder_prompts = [add_instructions_to_yprompt(new_q, bottleneck, template=self.y_template) 
                         for new_q, bottleneck in zip(new_question_texts, bottlenecks)]
        
        # 4) Call the decoder with the combined prompt to get the final output:
        decoder_outputs = self.decoder(decoder_prompts, yes_no_answer=True)

        # 5) Merge the final output with the extracted bottleneck (for inspection or debugging):
        final_outputs = []
        for decoder_output, raw_encoder, bottleneck in zip(decoder_outputs, raw_encoder_outputs, bottlenecks):
            final_output = dict(decoder_output)
            final_output["raw_encoder"] = raw_encoder
            final_output["bottleneck"] = bottleneck
            final_outputs.append(final_output)
        return final_outputs

    def __call_batch_multiple_y__(self, input_texts, new_question_texts):
        """
        Run the LBModel on the given input text:
          1) Use self.encoder to produce a textual summary in <info>...</info>.
          2) Extract the summary (bottleneck).
          3) For each of the new_questions, use self.decoder with (bottleneck + new_question_text).
          4) Return the decoder's output for each new question plus the 'bottleneck' for debugging.
        
        :param input_texts: List[str]
        :param new_question_texts: List[List[str]]
        :return: A dictionary typically including 'answer' and optional 'reasoning', 'logits', etc.
        """
        # 1) Produce a bottleneck summary from the encoder.
        # if cot, prompt to think step by step and lay out thinking process before final <info> summary
        encoder_prompts = [add_instructions_to_xprompt(input_text, self.encoder.use_cot, 
                                                     max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                     template=self.x_template) 
                          for input_text in input_texts]
        
        # encoder_output = self.encoder(encoder_prompt)
        args = {'max_new_tokens': self.encoder.max_cot_tokens} if type(self.encoder) != OpenAIModel else {}
        raw_encoder_outputs = self.encoder._generate(input_text=encoder_prompts, **args)
        
        # 2) Extract the <info> section from the encoder's output:
        bottlenecks = [self._extract_info_section(raw_encoder_output) for raw_encoder_output in raw_encoder_outputs]

        # 3) Iterate over each input and its corresponding questions
        final_outputs = []

        # Prepare all prompts and track counts for each bottleneck
        all_decoder_prompts = []
        counts = []

        for bottleneck, questions in zip(bottlenecks, new_question_texts):
            prompts = [add_instructions_to_yprompt(q, bottleneck, template=self.y_template) for q in questions]
            all_decoder_prompts.extend(prompts)
            counts.append(len(prompts))

        # Call decoder in batches (same batch size as encoder)
        all_decoder_outputs = []
        batch_size = len(input_texts)
        for i in range(0, len(all_decoder_prompts), batch_size):
            batch_prompts = all_decoder_prompts[i:i + batch_size]
            outputs = self.decoder(batch_prompts, yes_no_answer=True)
            all_decoder_outputs.extend(outputs)

        # Split outputs by bottleneck
        idx = 0
        for raw_encoder, bottleneck, count in zip(raw_encoder_outputs, bottlenecks, counts):
            outputs = all_decoder_outputs[idx:idx+count]
            final_outputs.append({
                "raw_encoder": raw_encoder,
                "bottleneck": bottleneck,
                "decoder_outputs": outputs
            })
            idx += count

        return final_outputs

    # total price property
    @property
    def total_price(self):
        return getattr(self.encoder, 'total_price', 0) + getattr(self.decoder, 'total_price', 0)
    
