from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, DistillationSetFitTrainer, sample_dataset
from setfit.exporters.onnx import export_onnx
import pickle

#https://github.com/huggingface/setfit/tree/main/src/setfit

def retrain_model(distilled = 0):
    """Retrains model, takes arg distilled: 0 for no 1 for yes"""
    
    global model_name

    match distilled:
        case "0":
            model_name = "regular"

            # Load a dataset from the Hugging Face Hub
            dataset = load_dataset("sst2")

            train_dataset = sample_dataset(dataset["train"],label_column="label", num_samples=8)
            eval_dataset = dataset["validation"]

            # Load a SetFit model from Hub
            model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

            # Create trainer
            trainer = SetFitTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss_class=CosineSimilarityLoss,
                metric="accuracy",
                batch_size=16,
                num_iterations=20, # The number of text pairs to generate for contrastive learning
                num_epochs=1, # perform a hyperparameter search on the number of epochs in the range [25,75] and pick the best performing model on a validation split
                column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
            )

            # Train and evaluate
            trainer.train()
            metrics = trainer.evaluate()
            message = "\nModel sucessfully retrained.\n"
            print("~" * len(message) + "~" * len(message))
            # print(trainer.model.model_head.classes_)
            # print(model.model_head.classes_)
        
        case 1:
            model_name = "distilled"

            # Load a dataset from the Hugging Face Hub
            dataset = load_dataset("sst2")

            train_dataset_teacher = sample_dataset(dataset["train"],label_column="label", num_samples=8)
            train_dataset_student = dataset["train"].shuffle(seed=0).select(range(500))
            eval_dataset = dataset["validation"]

            # Load a SetFit model from Hub
            teacher_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

            # Create trainer for teacher model
            teacher_trainer = SetFitTrainer(
                model=teacher_model,
                train_dataset=train_dataset_teacher,
                eval_dataset=eval_dataset,
                loss_class=CosineSimilarityLoss,
                metric="accuracy",
                batch_size=16,
                num_iterations=20, # The number of text pairs to generate for contrastive learning
                num_epochs=3, #A good rule of thumb is to start with a value that is 3 times the number of features in your data
                # Excerpt from the research paper: "...perform a hyperparameter search on the number of epochs in the range [25,75] and pick the best performing model on a validation split"
                column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
            )

            # Train and evaluate
            teacher_trainer.train()
            metrics = teacher_trainer.evaluate()


            ## MORE EPOCHS
            student_model = SetFitModel.from_pretrained("paraphrase-MiniLM-L3-v2")

            # Create trainer for knowledge distillation
            student_trainer = DistillationSetFitTrainer(
                teacher_model=teacher_model,
                train_dataset=train_dataset_student,
                student_model=student_model,
                eval_dataset=eval_dataset,
                loss_class=CosineSimilarityLoss,
                metric="accuracy",
                batch_size=16,
                num_iterations=20,
                num_epochs=3,
                column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
            )

            # Train student with knowledge distillation
            student_trainer.train()

            message = "\nModel sucessfully retrained.\n"
            print("~" * len(message) + "~" * len(message))
            # print(trainer.model.model_head.classes_)
            # print(model.model_head.classes_)
            model = student_model

            return model


def export_model(model, extension):
    """Exports the model. """

    match extension:
        case "pkl":    
            # Pickle
            pkl_filename = f"model/setfit_model_{model_name}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
            message = f"{model_name} model exported to pkl format.\n".capitalize()
            print("~" * len(message) + "\n" + message + "~" * len(message))

        case "onnx":
            # Export the sklearn based model to onnx
            output_path = f"model/setfit_model_{model_name}.onnx"
            export_onnx(model.model_body,
                        model.model_head,
                        opset=12,
                        output_path=output_path)
            message = f"{model_name} model exported to onnx format.\n".capitalize()
            print("~" * len(message) + "\n" + message + "~" * len(message))

        # case "setfitonnx":                        
        #     # Setfit Model directory
        #     model_path = "/model/setfit_model.pkl"
        #     # ONNX Output directory
        #     output_dir = "/model/setfit-onnx-lib-model.onnx"
        #     # Convert to ONNX
        #     convert_onnx(model_path=model_path,output_dir=output_dir)


export_model(retrain_model(1), "onnx")

