from src.models.model_loader import load_any_model
from src.models.lb_models.utils import add_instructions_to_yprompt


########## Ground-truth model
class LBModelGT:
    """
    Version of LBModel using ground-truth bottlenecks.
    """
    def __init__(
        self,
        decoder_config,
        x_template_file="base_x.yaml",
        y_template_file="base_y.yaml",
        **kwargs,
    ):
        """
        :param encoder_config: Configuration dictionary for the encoder (summarizer) model.
        :param decoder_config: Configuration dictionary for the decoder (final prediction) model.
        :param x_template_file: Filename for the encoder prompt template
        :param y_template_file: Filename for the decoder prompt template
        """
        self.decoder = load_any_model(decoder_config)
        self.students_df = None
        
        # Load templates
        from src.prompts.prompt_utils import load_prompt_template
        self.x_template = load_prompt_template(filename=x_template_file)
        self.y_template = load_prompt_template(filename=y_template_file)

    def set_students_df(self, students_df):
        """Set the students dataframe containing ground truth knowledge states."""
        self.students_df = students_df
        print('Set students_df for LBModelGT')

    def get_student_ks(self, user_id):
        dt_stud = self.students_df[self.students_df['student_id'] == int(user_id)].iloc[0]
        assert len(dt_stud) > 0, f"User ID {user_id} not found in students_df."
        return dt_stud.knowledge_state
    
    def __call__(self, new_question_text, user_ids):
        """
        Run the LBModelGT on the given new questions and user IDs:
          1) Get ground truth knowledge state for each user from students_df
          2) Use self.decoder with (knowledge state + new question)
          3) Return the decoder's output plus the 'bottleneck' for debugging.
        
        :param new_question_text: str or List[str]
        :param user_ids: List[str] or List[int]
        :return: A dictionary typically including 'answer' and optional 'reasoning', 'logits', etc.
        """
        assert self.students_df is not None, "students_df must be set before calling the model."
        if type(user_ids) == list:
            if type(new_question_text[0]) == list:
                return self.__call_batch_multiple_y__(new_question_text, user_ids)
            else:
                return self.__call_batch__(new_question_text, user_ids)
        else:
            if type(new_question_text) == list:
                return self.__call_batch_multiple_y__([new_question_text], [user_ids])
            
        # Get ground truth knowledge state for the user
        user_id = int(user_ids[0]) if isinstance(user_ids, list) else int(user_ids)
        knowledge_state = self.get_student_ks(user_id)

        # Construct a decoding prompt with the knowledge state
        decoder_prompt = add_instructions_to_yprompt(new_question_text, knowledge_state, 
                                                  template=self.y_template)
        
        # Call the decoder with the combined prompt to get the final output
        decoder_output = self.decoder(decoder_prompt, yes_no_answer=True)

        # Merge the final output with the knowledge state (for inspection or debugging)
        final_output = dict(decoder_output)  # make a copy
        final_output["raw_encoder"] = knowledge_state
        final_output["bottleneck"] = knowledge_state

        return final_output
    
    def __call_batch__(self, new_question_texts, user_ids):
        """
        Run the LBModelGT on a batch of new questions and user IDs.
        
        :param new_question_texts: List[str]
        :param user_ids: List[str] or List[int]
        :return: List of dictionaries with 'answer' and optional 'reasoning', 'logits', etc.
        """
        assert self.students_df is not None, "students_df must be set before calling the model."
        # Get ground truth knowledge states for all users
        knowledge_states = []
        for user_id in user_ids:
            knowledge_state = self.get_student_ks(user_id)
            knowledge_states.append(knowledge_state)

        # Construct decoding prompts with knowledge states
        decoder_prompts = [add_instructions_to_yprompt(new_q, ks, template=self.y_template) 
                          for new_q, ks in zip(new_question_texts, knowledge_states)]
        
        # Call the decoder with the combined prompts
        decoder_outputs = self.decoder(decoder_prompts, yes_no_answer=True)

        # Merge the final outputs with the knowledge states
        final_outputs = []
        for decoder_output, knowledge_state in zip(decoder_outputs, knowledge_states):
            final_output = dict(decoder_output)
            final_output["raw_encoder"] = knowledge_state
            final_output["bottleneck"] = knowledge_state
            final_outputs.append(final_output)
        return final_outputs

    def __call_batch_multiple_y__(self, new_question_texts, user_ids):
        """
        Run the LBModelGT on a batch of new questions and user IDs, where each user has multiple questions.
        
        :param new_question_texts: List[List[str]]
        :param user_ids: List[str] or List[int]
        :return: List of dictionaries with 'decoder_outputs' containing answers for each question
        """
        assert self.students_df is not None, "students_df must be set before calling the model."
        # Get ground truth knowledge states for all users
        knowledge_states = []
        for user_id in user_ids:
            knowledge_state = self.get_student_ks(user_id)
            knowledge_states.append(knowledge_state)

        # Prepare all prompts and track counts for each knowledge state
        all_decoder_prompts = []
        counts = []

        for knowledge_state, questions in zip(knowledge_states, new_question_texts):
            prompts = [add_instructions_to_yprompt(q, knowledge_state, template=self.y_template) 
                      for q in questions]
            all_decoder_prompts.extend(prompts)
            counts.append(len(prompts))

        # Call decoder once
        all_decoder_outputs = self.decoder(all_decoder_prompts, yes_no_answer=True)

        # Split outputs by knowledge state
        final_outputs = []
        idx = 0
        for knowledge_state, count in zip(knowledge_states, counts):
            outputs = all_decoder_outputs[idx:idx+count]
            final_outputs.append({
                "raw_encoder": knowledge_state,
                "bottleneck": knowledge_state,
                "decoder_outputs": outputs
            })
            idx += count

        return final_outputs
    
    # total price property
    @property
    def total_price(self):
        return getattr(self.decoder, 'total_price', 0)
    