import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import pickle
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


#https://github.com/huggingface/setfit/blob/main/notebooks/onnx_model_export.ipynb

input_text = ["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮", "I quit my job to look for new exciting opportunities"]
# look at embedding


## F1
@timer
def pickle_predict(input_text):
    # Run inference using the original model
    pkl_filename = "model/setfit_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    pytorch_preds = pickle_model(input_text)
    labeled_pkl_preds = list(map(labeler, pytorch_preds))

    return labeled_pkl_preds

@timer
def onnx_predict(input_text, outputh_path):
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


    output_path=outputh_path
    session = onnxruntime.InferenceSession(output_path)

    onnx_preds = session.run(None, dict(inputs))[0]

    labeled_onnx_preds = list(map(labeler, onnx_preds))

    # accuracy(session)

    # metrics = trainer.evaluate()

    # onnx_message = f'\nSetFit onnx model sentiment prediction: \n{labeled_onnx_preds}\n' 
    # print("~" * len(onnx_message) + onnx_message + "~" * len(onnx_message))

    # return {"prediction": labeled_onnx_preds, "time": time, "metrics": metrics }
    return labeled_onnx_preds


def labeler(preds):

    labeled_preds = ""

    match preds:
        case 0:
            labeled_preds = "Negative"
        case 1:
            labeled_preds = "Positive"
        case _:
            print ("Invalid prediction output.")

    return labeled_preds


pytorch_preds = pickle_predict(input_text)
onnx_preds = onnx_predict(input_text, "model/setfit_model.onnx")
distilled_onnx_preds = onnx_predict(input_text, "model/setfit_model_distilled.onnx")

@timer
def compare_pickle_onnx(pytorch_preds,onnx_preds, distilled_onnx_preds):
    """ Compare onnx and pkl predictions """
    pkl_message = f'\nPkl model prediction: {pytorch_preds}\n'
    onnx_message = f'\nOnnx model prediction: {onnx_preds}\n'
    distilled_onnx_message =  f'\nDistilled onnx model prediction: {distilled_onnx_preds}\n'
    print("~" * len(pkl_message) + pkl_message + "~" * len(pkl_message) + "\n" + "~" * len(onnx_message) + onnx_message + "~" * len(onnx_message)  + "\n" + "~" * len(distilled_onnx_message) + distilled_onnx_message + "~" * len(distilled_onnx_message))


# Compare onnx and pkl output
compare_pickle_onnx(pytorch_preds,onnx_preds, distilled_onnx_preds)