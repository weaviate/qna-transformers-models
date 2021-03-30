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
        inputs = self.tokenizer.encode_plus(input.question, input.text,
                add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]


        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        outputs = self.model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits


        print("start")
        print(answer_start_scores)
        print("end")
        print(answer_end_scores)

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        ids = input_ids[answer_start:answer_end]

        return self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(ids))
