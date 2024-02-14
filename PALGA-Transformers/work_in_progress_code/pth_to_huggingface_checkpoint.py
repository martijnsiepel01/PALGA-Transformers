import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration

config_path = '/home/msiepel/t5-small-config.json'
config = T5Config.from_json_file(config_path)


# Initialize the model with the configuration
model = AutoModelForSeq2SeqLM.from_config(config)

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/pretrained_models/epochs25_maxlen1024_trainbs8_valbs4_lr0.0001_datasetpretrain_all_patience3_commentpretrain_all_span_corruption_test.pth"))

model.save_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/flan-t5-pretrained-span-corruption")
