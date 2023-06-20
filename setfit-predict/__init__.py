import logging
from utils.predict import onnx_predict
from utils.load_file import load_model
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()

    if req_body:
        prediction = onnx_predict(load_model("setfit_model_distilled.onnx"), req_body)        
        return func.HttpResponse(f"This HTTP triggered function executed successfully./nSetfit prediction sample: {prediction[prediction]}")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass json in the request body for a prediction.",
             status_code=200
        )
