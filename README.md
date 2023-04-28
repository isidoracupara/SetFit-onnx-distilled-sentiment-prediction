# Publican-Be-SetFit-deployment
SetFit model for sentiment prediction in onnx format.
Functions are deployed on digital ocean and can be accessed through CLI. Example use:

Predict
```
curl -X POST "https://faas-ams3-2a2df116.doserverless.co/api/v1/web/fn-4d1f9416-5a80-46bb-8b00-bfbbbce69b76/default/SetFit_train" \
    -H "Content-Type: application/json" \
    -d {"input": ["i loved the spiderman movie!","pineapple on pizza is the worst 🤮"]}
```

Retrain
```
curl -X POST "https://faas-ams3-2a2df116.doserverless.co/api/v1/web/fn-4d1f9416-5a80-46bb-8b00-bfbbbce69b76/default/SetFit_train" \
    -H "Content-Type: application/json"

```