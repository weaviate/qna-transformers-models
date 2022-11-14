#!/usr/bin/env python3

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import os
import sys
import json

def quantization_config(onnx_cpu_arch: str):
    if onnx_cpu_arch.lower() == "avx512_vnni":
        print("getting quantization_config avx512_vnni")
        return AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    # defualt is ARM64
    return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

def fix_config_json(model_dir: str, model_name: str):
    with open(f"{model_dir}/config.json", 'r') as f:
        data = json.load(f)

    if model_name == "distilbert-base-uncased-distilled-squad":
        data["model_type"] = "distilbert"
    elif model_name == "distilbert-base-cased-distilled-squad":
        data["model_type"] = "distilbert"
    elif model_name == "bert-large-uncased-whole-word-masking-finetuned-squad":
        data["model_type"] = "bert"
    elif model_name == "deepset/bert-large-uncased-whole-word-masking-squad2":
        data["model_type"] = "bert"

    with open(f"{model_dir}/config.json", 'w') as json_file:
        json.dump(data, json_file)

model_dir = './models/model'
model_name = os.getenv('MODEL_NAME')
if model_name is None or model_name == "":
    print("Fatal: MODEL_NAME is required")
    sys.exit(1)

onnx_runtime = os.getenv('ONNX_RUNTIME')
if not onnx_runtime:
    onnx_runtime = "false"

onnx_cpu_arch = os.getenv('ONNX_CPU')
if not onnx_cpu_arch:
    onnx_cpu_arch = "ARM64"

print("Downloading model {} from huggingface model hub, onnx_runtime: {}, onnx_cpu_arch: {}".format(model_name, onnx_runtime, onnx_cpu_arch))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_dir)

if onnx_runtime.lower() == "true" or onnx_runtime == "1":
    # Download the model
    ort_model = ORTModelForQuestionAnswering.from_pretrained(model_name, from_transformers=True)
    ort_model.save_pretrained(model_dir)
    # Quantize the model / convert to ONNX
    qconfig = quantization_config(onnx_cpu_arch)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    # Apply dynamic quantization on the model
    quantizer.quantize(save_dir=model_dir, quantization_config=qconfig)
    fix_config_json(model_dir, model_name)
else:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(model_dir)
