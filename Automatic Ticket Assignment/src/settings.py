import pathlib
import os

project_root = str(pathlib.Path(__file__).parent.parent.absolute())
data_path = os.path.join(project_root, "data/")
saved_models_path = os.path.join(project_root, "src/", "saved_models/")

settings = {
    "project_root": project_root,
    "data_path": data_path,
    "saved_models_path": saved_models_path,
    "vocabulary_count": 20000,
    "active_model_key": "BIDIRECTIONAL_LSTM",
    "models": {
        "BIDIRECTIONAL_GRU": {
            "model_key": "BIDIRECTIONAL_GRU",
            "description":  "basic GRU implementation",
            "model_path": "src.nlp_models.bidirectional_gru_v1",
            "train_samples": 1000,
            "test_samples": 500,
            "model_compile_options": {
                "embedding_dimensions": 150,
                "loss":"categorical_crossentropy",
                "optimizer": "adam",
                "metrics":['accuracy'],
                "DESC_MAX_WORDS": "DESC_MAX_WORDS",
                "SHORT_DESC_MAX_WORDS": "SHORT_DESC_MAX_WORDS",
                "TARGET_LEN": "TARGET_LEN",
                "tokenizer": "tokenizer",
                },
            "model_training_options": {
                "train_input_x": ["x_train", "x_train_short"],
                "train_input_y": "y_train"
            },
            "model_evaluation_options": {
                "test_input_x": ["x_test", "x_test_short"],
                "test_input_y": "y_test"
            },
            "weights_path": "",
        }
    },
    "word_index_path": "",
    "DB": {
        "connection_type": "MONGO",
        "uri": "mongodb://user:pass@host:port/db"
    },
    "max_sentence_length": 250,
    "training": {
        "vocabulary_count": 20000,
        "word_to_index_method": "tfidf or counter",
    }
}


if __name__ == "__main__":
    print("settings: ", settings)
