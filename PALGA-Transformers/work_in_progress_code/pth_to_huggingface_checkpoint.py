import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration

config_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/mT5_base/config.json'
config = MT5Config.from_json_file(config_path)


# Initialize the model with the configuration

model = MT5ForConditionalGeneration(config=config)

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
tokenizer = MT5Tokenizer.from_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/flan_tokenizer")
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs40_data_sethisto_commentmt5_base_baseline_with_codes_codesTrue.pth"))

model.save_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/mT5_base_baseline_checkpoint")
