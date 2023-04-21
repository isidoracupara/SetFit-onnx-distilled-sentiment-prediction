import os
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import pickle

#https://github.com/huggingface/setfit/blob/main/notebooks/onnx_model_export.ipynb

# Run inference using the original model
input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]
pkl_filename = "model/setfit_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

pytorch_preds = pickle_model(input_text)
print('pkl model prediction: ', pytorch_preds)

# Run inference using onnx model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
inputs = tokenizer(
    input_text,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_tensors="np",
)

output_path="model/setfit_model.onnx"
session = onnxruntime.InferenceSession(output_path)

onnx_preds = session.run(None, dict(inputs))[0]
print('onnx model prediction: ', onnx_preds)

assert np.array_equal(onnx_preds, pytorch_preds)