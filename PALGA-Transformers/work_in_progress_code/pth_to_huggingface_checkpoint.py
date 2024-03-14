import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration

# config_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5-small/config.json'
# config = T5Config.from_json_file(config_path)


# Initialize the model with the configuration

# model = MT5ForConditionalGeneration(config=config)
tokenizer = T5Tokenizer.from_pretrained("google/mT5-small")

config = T5Config(decoder_start_token_id=tokenizer.pad_token_id) 
model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/pretrained_models/num_train_epochs5_data_setpretrain_commenttest_taskspan_corruption.pth"))

model.save_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5_small_pretrain_span_corruption_20_percent")
