import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration, AutoTokenizer

# config_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5-small/config.json'
# config = T5Config.from_json_file(config_path)


# Initialize the model with the configuration

# model = MT5ForConditionalGeneration(config=config)
# tokenizer = AutoTokenizer.from_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_with_codes_csep_normal_token")
tokenizer = AutoTokenizer.from_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/T5_small_32128_pretrain_with_codes")
print(len(tokenizer))

config = T5Config(decoder_start_token_id=tokenizer.pad_token_id) 
model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/pretrained_models/num_train_epochs50_data_setpretrain_commentpretrain_100_percent_custom_tokenizer_with_full_codes_taskspan_corruption.pth"))

model.save_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/T5-small_v2")
