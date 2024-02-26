import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer, T5Tokenizer, T5ForConditionalGeneration

config_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json'
config = T5Config.from_json_file(config_path)


# Initialize the model with the configuration

model = AutoModelForSeq2SeqLM.from_config(config)

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
tokenizer = T5Tokenizer("/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/combined_data_unigram_t5_custom_16000_1_lower_case.model")
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_datasetall_modelflan-t5-small_commentcombined_data_unigram_lower_case_16000_1_combined_data_patience3_freezeallbutxlayers0_lrstrategyAdamW.pth"))

model.save_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/unigram_16000_1_lower_case_checkpoint")
