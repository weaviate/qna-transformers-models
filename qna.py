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

    def tryToAnswer(self, inputs_question, inputs_text):
        input_ids_question = inputs_question["input_ids"].tolist()[0]
        input_ids_text = inputs_text["input_ids"].tolist()[0]

        input_ids = input_ids_question + input_ids_text

        inputs = {
            'input_ids': torch.cat((inputs_question['input_ids'], inputs_text['input_ids']), 1),
            'token_type_ids': torch.cat((inputs_question['token_type_ids'], inputs_text['token_type_ids']), 1),
            'attention_mask': torch.cat((inputs_question['attention_mask'], inputs_text['attention_mask']), 1),
        }

        outputs = self.model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        start_score = torch.max(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        end_score = torch.max(answer_end_scores)

        score = (start_score + end_score) / 2
        ids = input_ids[answer_start:answer_end]
        certainty = float(score) / 10

        return self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(ids)), certainty

    def getInputsText(self, inputs_text, window_start, window_end):
        new_inputs_text={
                'input_ids': torch.tensor([inputs_text['input_ids'][0][window_start:window_end].tolist() + [101]]),
                'token_type_ids': torch.tensor([inputs_text['token_type_ids'][0][window_start:window_end].tolist() + [0]]),
                'attention_mask': torch.tensor([inputs_text['attention_mask'][0][window_start:window_end].tolist() + [1]]),
        }
        return new_inputs_text

    def getTokenizedInputs(self, question, text):
        inputs_question = self.tokenizer('[CLS] ' + question + ' [SEP] ', add_special_tokens=False, 
                return_tensors="pt")

        inputs_text = self.tokenizer(text + ' [CLS]', add_special_tokens=False, 
                return_tensors="pt")

        total_length = len(inputs_question['input_ids'][0]) + len(inputs_text['input_ids'][0])

        def windowSlice(arr, start, stop, block, overlap, arr_to_cut, result=[]):
            if start == len(arr):
                return
            if stop > len(arr):
                result.append(self.getInputsText(arr_to_cut, start, len(arr)))
                return
            else:
                result.append(self.getInputsText(arr_to_cut, start, stop))
            windowSlice(arr, stop-overlap, stop-overlap+block, block, overlap, arr_to_cut, result)

        treshold = 512
        windowed_input_texts=[]
        if total_length > treshold:
            block = treshold - len(inputs_question['input_ids'][0]) - 1
            overlap = int(int(block) / 2)
            windowSlice(inputs_text['input_ids'][0], 0, block, block, overlap, inputs_text, windowed_input_texts)
        else:
            windowed_input_texts.append(inputs_text)

        result = []
        for window_input_text in windowed_input_texts:
            result.append((inputs_question, window_input_text))

        return result

    # TODO: what do we need to do to support CUDA?
    async def do(self, input: AnswersInput):
        question = input.question
        text = input.text
        if len(question) == 0 or len(text) == 0:
            return None, None

        answersScores=[]
        inputs = self.getTokenizedInputs(question, text)
        for input_value in inputs:
            answersScores.append(self.tryToAnswer(input_value[0], input_value[1]))

        response = ("", 0.0)
        if len(answersScores) > 0:
            for res in answersScores:
                if res[1] > response[1]:
                    response = res
                # print("answer: ", res[0], res[1])

        answer = response[0]
        certainty = response[1]
        if len(answer) > 0 and (answer.find("[CLS]") != -1 or answer.find("[SEP]") != -1):
            return None, None

        return answer, certainty
