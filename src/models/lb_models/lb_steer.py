from src.models.lb_models.lb_base import LBModel
from src.models.api_models import OpenAIModel
from src.data.utils_iterative import batch_to_history_text
from src.models.lb_models.utils import add_instructions_to_xprompt, add_instructions_to_yprompt
import random

########## Steer model
class LBModelSteer(LBModel):
    def __init__(self, 
                 filter_cid=None, 
                 additional_x_info=None, 
                 additional_bn_info=None,
                 q_per_batch=50, 
                 x_template_file="base_x.yaml",
                 y_template_file="base_y.yaml",
                 student_dict=None,
                 **kwargs):
        super().__init__(x_template_file=x_template_file, 
                         y_template_file=y_template_file, 
                         **kwargs)
        self.filter_cid = filter_cid
        self.additional_x_info = additional_x_info
        self.additional_bn_info = additional_bn_info
        self.q_per_batch = q_per_batch
        self.contains = None
        self.student_dict = student_dict if student_dict else {}

    # API ----------------------------------------------------------
    def set_contains(self, contains):
        self.contains = contains
        
    def set_student_dict(self, student_dict):
        """Set the student dictionary containing student objects."""
        self.student_dict = student_dict
        print('Set student_dict for LBModelSteer')
        
    def _get_student_misc(self, user_id, n_misc):
        """Extract a random misconception text from the student.
        
        Args:
            user_id: The student ID
            
        Returns:
            str: A misconception description or empty string if student has none
        """
        student = self.student_dict.get(int(user_id))
        if not student or not student.misconceptions:
            return ""
        
        # Choose a random misconception
        # random_misc = random.choice(student.misconceptions)
        # pick n random misconceptions without replacement
        random_misconceptions = random.sample(student.misconceptions, min(n_misc, len(student.misconceptions)))
        # Join them into a single string
        misc_str = "\n".join([f"{misc.description}" for misc in random_misconceptions])
        return misc_str

    
    def _get_student_mastery(self, user_id, construct_id):
        """Extract whether the student has mastered a particular construct.
        
        Args:
            construct_id: The ID of the construct to check mastery for
            
        Returns:
            str: Description of the student's mastery level for the construct
        """
        if not user_id:
            raise ValueError("user_id is required to get student mastery.")
            
        student = self.student_dict.get(int(user_id))
        if not student:
            raise ValueError(f"Student with ID {user_id} not found.")
            
        from src.synthetic_dataset.constructs import Construct
        # Convert construct_id to Construct enum
        # try:
        construct_id = int(construct_id)
        construct = list(Construct)[construct_id]
        mastery = student.skill_mastery.get(construct, 0.0)
        if mastery == 1.0:
            return f"The student has mastered {construct.value} except in the event of misconceptions."
        else:
            return f"The student always fails with {construct.value}."
        # except (ValueError, IndexError):
            # return ""

    def _process_dynamic_info(self, info_str, user_id):
        """Process dynamic information strings like $get_student_misc:1
        
        Args:
            info_str: The information string that may contain dynamic elements
            user_id: The user ID to use for this processing
            
        Returns:
            str: The processed information string
        """
        if not info_str or not isinstance(info_str, str):
            return info_str
            
        # Check for dynamic elements
        if "$get_student_misc:" in info_str:
            # Extract the user ID from the string
            parts = info_str.split("$get_student_misc:")
            n_misc = int(parts[1].split()[-1])
            misc_info = self._get_student_misc(user_id=user_id, n_misc=n_misc)
            if misc_info:
                return f"The student has at least the following misconception(s):\n{misc_info}"
            else:
                return "The student has no known misconceptions."
                
        elif "$get_student_mastery:" in info_str:
            # Extract the construct ID from the string
            parts = info_str.split("$get_student_mastery:")
            construct_id = int(parts[1].split()[-1])
            return self._get_student_mastery(user_id=user_id, construct_id=construct_id)
                
        return info_str

    def _filter_questions(self, input_data):
        """Filter out questions with construct_id matching self.filter_cid"""
        if self.filter_cid is None:
            return input_data

        # Handle both single trajectory and batch of trajectories
        if isinstance(input_data, list):
            if len(input_data) == 0:
                return input_data
            
            # Check if we have a batch of trajectories or a single trajectory
            if isinstance(input_data[0], list):
                # Batch of trajectories: List[List[Dict]]
                return [[entry for entry in traj if entry.get("construct_id") != self.filter_cid] 
                        for traj in input_data]
            else:
                # Single trajectory: List[Dict]
                return [entry for entry in input_data if entry.get("construct_id") != self.filter_cid]
        
        return input_data

    def __call__(self, input_text, new_question_text, user_id=None):
        if isinstance(input_text, list) and isinstance(input_text[0], list):
            assert type(new_question_text) == list
            if type(new_question_text[0]) == list:
                return self.__call_batch_multiple_y__(input_text, new_question_text, user_ids=user_id)
            else:
                return self.__call_batch__(input_text, new_question_text, user_ids=user_id)

        # ---------- steer path --------------------------------
        if isinstance(input_text, list) and isinstance(input_text[0], dict):
            assert self.contains is not None, "`contains` not set"
            
            # Filter questions if needed
            filtered_input = self._filter_questions(input_text)
            
            # Convert input to text
            qa_block = batch_to_history_text(filtered_input, self.contains)
            
            # Add additional x info if provided, process dynamic info if needed
            additional_x_info = self.additional_x_info
            if additional_x_info and user_id is not None:
                additional_x_info = self._process_dynamic_info(additional_x_info, user_id)
            
            if additional_x_info:
                qa_block = f"{qa_block}\n\n{additional_x_info}"
            
            # Build encoder prompt
            encoder_prompt = add_instructions_to_xprompt(qa_block, self.encoder.use_cot, 
                                                        max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                        template=self.x_template)
            
            # Generate bottleneck
            args = {'max_new_tokens': self.encoder.max_cot_tokens} if type(self.encoder) != OpenAIModel else {}
            raw_encoder = self.encoder._generate(input_text=encoder_prompt, **args)
            
            # Extract bottleneck
            bottleneck = self._extract_info_section(raw_encoder)
            
            # Add additional bottleneck info if provided, process dynamic info if needed
            additional_bn_info = self.additional_bn_info
            if additional_bn_info and user_id is not None:
                additional_bn_info = self._process_dynamic_info(additional_bn_info, user_id)
                
            if additional_bn_info:
                bottleneck = f"{bottleneck}\n\n{additional_bn_info}"
            
            # Generate decoder prompt
            decoder_prompt = add_instructions_to_yprompt(new_question_text, bottleneck, 
                                                        template=self.y_template)
            
            # Call decoder
            decoder_output = self.decoder(decoder_prompt, yes_no_answer=True)
            
            # Prepare final output
            final_output = dict(decoder_output)
            final_output["raw_encoder"] = raw_encoder
            final_output["bottleneck"] = bottleneck
            
            return final_output
        
        else:
            # Fall back to single-shot behaviour
            return super().__call__(input_text, new_question_text)

    def __call_batch__(self, input_texts, new_question_texts, user_ids=None):
        assert self.contains is not None, "`contains` not set"
        assert len(input_texts) == len(new_question_texts)
        if user_ids is not None:
            assert len(input_texts) == len(user_ids), "Number of user IDs must match number of inputs"
        
        # Filter questions if needed
        filtered_inputs = self._filter_questions(input_texts)
        
        # Convert inputs to text
        qa_blocks = [batch_to_history_text(filtered_input, self.contains) for filtered_input in filtered_inputs]
        
        # Add additional x info if provided
        if self.additional_x_info:
            if user_ids is not None:
                # Process dynamic information for each user
                for i, (qa_block, user_id) in enumerate(zip(qa_blocks, user_ids)):
                    additional_x_info = self._process_dynamic_info(self.additional_x_info, user_id)
                    qa_blocks[i] = f"{qa_block}\n\n{additional_x_info}"
            else:
                qa_blocks = [f"{qa_block}\n\n{self.additional_x_info}" for qa_block in qa_blocks]
        
        # Build encoder prompts
        encoder_prompts = [add_instructions_to_xprompt(qa_block, self.encoder.use_cot, 
                                                    max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                    template=self.x_template) 
                        for qa_block in qa_blocks]
        
        # Generate bottlenecks
        args = {'max_new_tokens': self.encoder.max_cot_tokens} if type(self.encoder) != OpenAIModel else {}
        raw_encoders = self.encoder._generate(input_text=encoder_prompts, **args)
        if isinstance(raw_encoders, str):
            raw_encoders = [raw_encoders]
        
        # Extract bottlenecks
        bottlenecks = [self._extract_info_section(raw_encoder) for raw_encoder in raw_encoders]
        
        # Add additional bottleneck info if provided
        if self.additional_bn_info:
            if user_ids is not None:
                # Process dynamic information for each user
                for i, (bn, user_id) in enumerate(zip(bottlenecks, user_ids)):
                    additional_bn_info = self._process_dynamic_info(self.additional_bn_info, user_id)
                    bottlenecks[i] = f"{bn}\n\n{additional_bn_info}"
            else:
                bottlenecks = [f"{bn}\n\n{self.additional_bn_info}" for bn in bottlenecks]
        
        # Generate decoder prompts
        decoder_prompts = [add_instructions_to_yprompt(new_q, bottleneck, template=self.y_template) 
                        for new_q, bottleneck in zip(new_question_texts, bottlenecks)]
        
        # Call decoder
        decoder_outputs = self.decoder(decoder_prompts, yes_no_answer=True)
        
        # Prepare final outputs
        final_outputs = []
        for encoder_prompt, decoder_output, raw_encoder, bottleneck in zip(encoder_prompts, decoder_outputs, raw_encoders, bottlenecks):
            final_output = dict(decoder_output)
            final_output["encoder_prompt"] = encoder_prompt
            final_output["raw_encoder"] = raw_encoder
            final_output["bottleneck"] = bottleneck
            final_outputs.append(final_output)
        
        return final_outputs

    def __call_batch_multiple_y__(self, input_texts, new_question_texts, user_ids=None):
        assert self.contains is not None, "`contains` not set"
        assert len(input_texts) == len(new_question_texts)
        if user_ids is not None:
            assert len(input_texts) == len(user_ids), "Number of user IDs must match number of inputs"
        
        # Filter questions if needed
        filtered_inputs = self._filter_questions(input_texts)
        
        # Convert inputs to text
        qa_blocks = [batch_to_history_text(filtered_input, self.contains) for filtered_input in filtered_inputs]
        
        # Add additional x info if provided
        if self.additional_x_info:
            if user_ids is not None:
                # Process dynamic information for each user
                for i, (qa_block, user_id) in enumerate(zip(qa_blocks, user_ids)):
                    additional_x_info = self._process_dynamic_info(self.additional_x_info, user_id)
                    qa_blocks[i] = f"{qa_block}\n\n{additional_x_info}"
            else:
                qa_blocks = [f"{qa_block}\n\n{self.additional_x_info}" for qa_block in qa_blocks]
        
        # Build encoder prompts
        encoder_prompts = [add_instructions_to_xprompt(qa_block, self.encoder.use_cot, 
                                                    max_bottleneck_tokens=self.max_bottleneck_tokens,
                                                    template=self.x_template) 
                        for qa_block in qa_blocks]
        
        # Generate bottlenecks
        args = {'max_new_tokens': self.encoder.max_cot_tokens} if type(self.encoder) != OpenAIModel else {}
        raw_encoders = self.encoder._generate(input_text=encoder_prompts, **args)
        if isinstance(raw_encoders, str):
            raw_encoders = [raw_encoders]
        
        # Extract bottlenecks
        bottlenecks = [self._extract_info_section(raw_encoder) for raw_encoder in raw_encoders]
        
        # Add additional bottleneck info if provided
        if self.additional_bn_info:
            if user_ids is not None:
                # Process dynamic information for each user
                for i, (bn, user_id) in enumerate(zip(bottlenecks, user_ids)):
                    additional_bn_info = self._process_dynamic_info(self.additional_bn_info, user_id)
                    bottlenecks[i] = f"{bn}\n\n{additional_bn_info}"
            else:
                bottlenecks = [f"{bn}\n\n{self.additional_bn_info}" for bn in bottlenecks]
        
        # Prepare all prompts and track counts for each bottleneck
        all_decoder_prompts = []
        counts = []
        
        for bottleneck, questions in zip(bottlenecks, new_question_texts):
            prompts = [add_instructions_to_yprompt(q, bottleneck, template=self.y_template) for q in questions]
            all_decoder_prompts.extend(prompts)
            counts.append(len(prompts))
        
        # Call decoder in batches
        all_decoder_outputs = []
        batch_size = len(input_texts)
        for i in range(0, len(all_decoder_prompts), batch_size):
            batch_prompts = all_decoder_prompts[i:i + batch_size]
            outputs = self.decoder(batch_prompts, yes_no_answer=True)
            all_decoder_outputs.extend(outputs)
        
        # Split outputs by bottleneck
        final_outputs = []
        idx = 0
        for encoder_prompt, raw_encoder, bottleneck, count in zip(encoder_prompts, raw_encoders, bottlenecks, counts):
            outputs = all_decoder_outputs[idx:idx+count]
            final_outputs.append({
                "encoder_prompt": encoder_prompt,
                "raw_encoder": raw_encoder,
                "bottleneck": bottleneck,
                "decoder_outputs": outputs
            })
            idx += count
        
        return final_outputs

