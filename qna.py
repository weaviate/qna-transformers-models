from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from pydantic import BaseModel
from typing import Optional
import torch


class AnswersInput(BaseModel):
    text: str
    question: str


class Qna:
    model: AutoModelForQuestionAnswering
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        if self.cuda:
            self.model.to(self.cuda_core)
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # TODO: what do we need to do to support CUDA?
    async def do(self, input: AnswersInput):

        inputs_question = self.tokenizer('[CLS] ' + input.question + ' [SEP] ' , add_special_tokens=False, 
                return_tensors="pt")

        inputs_text = self.tokenizer(input.text + ' [CLS]', add_special_tokens=False, 
                return_tensors="pt")

        total_length = len(inputs_question['input_ids'][0]) + len(inputs_text['input_ids'][0])
        print("total lenght is {}".format(total_length))

        window_start=100
        window_end=200
        if total_length > 256:
            inputs_text={
                    'input_ids': torch.tensor([inputs_text['input_ids'][0][window_start:window_end].tolist() + [101]]),
                    'token_type_ids': torch.tensor([inputs_text['token_type_ids'][0][window_start:window_end].tolist() + [0]]),
                    'attention_mask': torch.tensor([inputs_text['attention_mask'][0][window_start:window_end].tolist() + [1]]),
            }



        input_ids_question = inputs_question["input_ids"].tolist()[0]
        input_ids_text = inputs_text["input_ids"].tolist()[0]

        input_ids = input_ids_question + input_ids_text


        inputs = {
            'input_ids': torch.cat((inputs_question['input_ids'], inputs_text['input_ids']), 1),
            'token_type_ids': torch.cat((inputs_question['token_type_ids'], inputs_text['token_type_ids']), 1),
            'attention_mask': torch.cat((inputs_question['attention_mask'], inputs_text['attention_mask']), 1),
        }
        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        print(text_tokens)
        outputs = self.model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits


        print("start")
        print(answer_start_scores)
        print("end")
        print(answer_end_scores)

        answer_start = torch.argmax(answer_start_scores)
        start_score = torch.max(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        end_score = torch.max(answer_end_scores)

        score = (start_score + end_score) / 2
        print("score is {}".format(score))
        ids = input_ids[answer_start:answer_end]

        return self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(ids))
