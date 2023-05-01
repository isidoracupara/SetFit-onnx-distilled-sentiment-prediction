import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import pickle

#https://github.com/huggingface/setfit/blob/main/notebooks/onnx_model_export.ipynb

input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"]


def pickle_predict(input_text):
    # Run inference using the original model
    pkl_filename = "model/setfit_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    pytorch_preds = pickle_model(input_text)
    return pytorch_preds

pytorch_preds = pickle_predict(input_text)


def onnx_predict(input_text):
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

    onnx_message = '\nSetFit onnx model sentiment prediction: \n{onnx_preds}\n' 
    print("~" * len(onnx_message) + onnx_message + "~" * len(onnx_message))

    return onnx_preds


onnx_preds = onnx_predict(input_text)


def compare_pickle_onnx(pytorch_preds,onnx_preds):
    pkl_message = '\nPkl model prediction: {pytorch_preds}\n'
    onnx_message = '\nOnnx model prediction: {onnx_preds}\n' 
    print("~" * len(pkl_message) + pkl_message + onnx_message + "~" * len(onnx_message))
    assert np.array_equal(onnx_preds, pytorch_preds)


compare_pickle_onnx()