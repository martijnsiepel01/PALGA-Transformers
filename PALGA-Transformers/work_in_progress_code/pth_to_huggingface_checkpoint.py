import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration

config_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5_small/config.json'
config = MT5Config.from_json_file(config_path)


# Initialize the model with the configuration

model = MT5ForConditionalGeneration(config=config)

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mT5-small")
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentmt5_small_all_with_codes_long.pth"))

model.save_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5_small_codes_long_checkpoint")
