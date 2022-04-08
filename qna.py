from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
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


        inputs = BatchEncoding({
            'input_ids': torch.cat((inputs_question['input_ids'], inputs_text['input_ids']), 1),
            'attention_mask': torch.cat((inputs_question['attention_mask'], inputs_text['attention_mask']), 1),
        })

        if 'token_type_ids' in inputs_text:
            # increase token_type_ids of the second part as that is what would
            # have happened naturally if they had been tokenized in one go
            inputs_text['token_type_ids'] = torch.add(inputs_text['token_type_ids'], 1)

            # and splice them together
            inputs['token_type_ids'] = torch.cat((inputs_question['token_type_ids'], inputs_text['token_type_ids']), 1)

        outputs = self.model(**inputs)

        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        start_score = torch.max(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        end_score = torch.max(answer_end_scores)

        score = (start_score + end_score) / 2
        ids = input_ids[answer_start:answer_end]
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids))
        certainty = 0.0
        if self.isGoodAnswer(answer):
            certainty = float(score) / 10

        return answer, certainty

    def getInputsText(self, inputs_text, window_start, window_end):
        new_inputs_text=BatchEncoding({
                'input_ids': torch.tensor([inputs_text['input_ids'][0][window_start:window_end].tolist() + [101]]),
                'attention_mask': torch.tensor([inputs_text['attention_mask'][0][window_start:window_end].tolist() + [1]]),
        })
        if 'token_type_ids' in inputs_text:
            new_inputs_text['token_type_ids'] = torch.tensor([inputs_text['token_type_ids'][0][window_start:window_end].tolist() + [0]])
        return new_inputs_text

    def getTokenizedInputs(self, question, text):
        inputs_question = self.tokenizer('[CLS] ' + question + ' [SEP] ', add_special_tokens=False, 
                return_tensors="pt")

        inputs_text = self.tokenizer(text + ' [SEP]', add_special_tokens=False, 
                return_tensors="pt")

        windowed_input_texts = []
        input_texts_arr = self.performWindowSliceOnInput(inputs_text)
        for input_text in input_texts_arr:
            total_length = len(inputs_question['input_ids'][0]) + len(input_text['input_ids'][0])
            res = self.performWindowSliceToFindAnswer(inputs_question, input_text, total_length)
            windowed_input_texts.extend(res)

        result = []
        for window_input_text in windowed_input_texts:
            if self.cuda:
                inputs_question.to(self.cuda_core)
                window_input_text.to(self.cuda_core)
            result.append((inputs_question, window_input_text))

        return result

    def performWindowSliceOnInput(self, inputs_text):
        # this method cuts very long tests in order to prevent maximum recursion depth exceeded error
        # cuts the input without overlap and without taking into account questions length
        # the cutoff is set to 220k elements
        return self.performWindowSlice(None, inputs_text, len(inputs_text['input_ids'][0]), 220000, False)

    def performWindowSliceToFindAnswer(self, inputs_question, inputs_text, total_length):
        return self.performWindowSlice(inputs_question, inputs_text, total_length, 512, True)

    def performWindowSlice(self, inputs_question, inputs_text, total_length, treshold, withOverlap):
        def windowSlice(len_arr, start, block, overlap, arr_to_cut, result=[]):
            if start + block >= len_arr:
                result.append(self.getInputsText(arr_to_cut, start, len_arr))
                return
            else:
                result.append(self.getInputsText(arr_to_cut, start, start + block))
            windowSlice(len_arr, start + block-overlap, block, overlap, arr_to_cut, result)

        windowed_input_texts=[]
        if total_length > treshold:
            block = treshold - 1
            if inputs_question is not None:
                block = block - len(inputs_question['input_ids'][0]) # take into account question length also
            overlap = 0
            if withOverlap is True:
                overlap = int(int(block) / 4) # 25% overlap set
            windowSlice(len(inputs_text['input_ids'][0]), 0, block, overlap, inputs_text, windowed_input_texts)
        else:
            windowed_input_texts.append(inputs_text)

        return windowed_input_texts

    def isGoodAnswer(self, answer):
        return answer != None and len(answer) > 0 and not (answer.find("[CLS]") != -1 or answer.find("[SEP]") != -1)

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
        if not self.isGoodAnswer(answer):
            return None, None

        return answer, certainty
