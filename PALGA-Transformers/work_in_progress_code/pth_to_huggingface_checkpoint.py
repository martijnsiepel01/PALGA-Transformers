import torch
from transformers import AutoModelForSeq2SeqLM, T5Config, MT5Tokenizer

config_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json'
config = T5Config.from_json_file(config_path)


# Initialize the model with the configuration
model = AutoModelForSeq2SeqLM.from_config(config)

tokenizer = MT5Tokenizer(vocab_file='/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/unigram_t5_custom_16000_1.model')
model.resize_token_embeddings(len(tokenizer))

# Load the weights
model.load_state_dict(torch.load("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentunigram_custom_tokenizer_vocab_16000_098_patience3_freezeallbutxlayers0.pth"))

model.save_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/flan-small-unigram-098-checkpoint")
