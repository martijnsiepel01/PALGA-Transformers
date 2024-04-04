from transformers import MT5ForConditionalGeneration, MT5Config, MT5Tokenizer, AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config, T5Tokenizer, AutoConfig
import torch
from datasets import load_dataset
import random
import pandas as pd


# Path to your PyTorch checkpoint (.pth) file
checkpoint_model1 = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs25_data_setall_commentmT5_small_pretrained_v1_all.pth'
# checkpoint_model2 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentunigram_custom_tokenizer_vocab_16000_1_patience3_freezeallbutxlayers0.pth'
# checkpoint_model3 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentcustom_tokenizer_vocab_16000_patience3_freezeallbutxlayers0.pth'
# checkpoint_model4 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentcustom_tokenizer_vocab_16000_098_patience3_freezeallbutxlayers0.pth'

# # Define the configuration for your MT5 model
# config = T5Config.from_pretrained('/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5-small/config.json')
config = AutoConfig.from_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5_small_v1", local_files_only=True)


# tokenizer1 = MT5Tokenizer.from_pretrained('google/mT5-small')
tokenizer1 = AutoTokenizer.from_pretrained("/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_with_codes_csep_normal_token")
# tokenizer2 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/unigram_t5_custom_16000_1.model')
# tokenizer3 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/t5_custom_16000_1.model')
# tokenizer4 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/t5_custom_16000_098.model')


# # config = MT5Config.from_pretrained("google/flan-t5-small")
# # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# max_length_sentence = 2048

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = examples["Conclusie"]
    targets = examples["Codes"]
    # Since we're dealing with single examples, no need to iterate
    model_inputs = tokenizer(inputs, max_length=max_length_sentence, truncation=True, padding="max_length", return_tensors="pt")

    # We use tokenizer.encode to handle the target, ensuring it's treated as a single sequence
    # Note: This assumes 'targets' is a single string or None. Adjust accordingly if it's not the expected format.
    if targets:  # Ensure there is a target to encode
        labels = tokenizer(targets, max_length=max_length_sentence, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
    else:
        labels = None

    model_inputs["labels"] = labels
    return model_inputs

def prepare_test_dataset(test_data_location, tokenizer, max_length_sentence):
    # test_data_location = "/home/msiepel/data/all/all_norm_test.tsv"
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    # tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    test_dataset = tokenized_datasets['train']
    return test_dataset


# test_dataset_1 = prepare_test_dataset('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv', tokenizer1, max_length_sentence)
# test_dataset_2 = prepare_test_dataset('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_P1.tsv', tokenizer2, max_length_sentence)

model1 = T5ForConditionalGeneration(config)
model1.resize_token_embeddings(len(tokenizer1))

# model2 = T5ForConditionalGeneration(config)
# model2.resize_token_embeddings(len(tokenizer2))

# model3 = T5ForConditionalGeneration(config)
# model3.resize_token_embeddings(len(tokenizer3))

# model4 = T5ForConditionalGeneration(config)
# model4.resize_token_embeddings(len(tokenizer4))

thesaurus_location = '/home/msiepel/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin')

def get_word_from_code(code):
    if code == '[c-sep]' or code == '[C-SEP]':
        return '[C-SEP]'
    else:
        word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
        return word[0] if len(word) > 0 else 'Unknown'


state_dict1 = torch.load(checkpoint_model1, map_location='cpu')  # 'map_location' argument for loading on CPU
model1.load_state_dict(state_dict1)

input_text = '''I. Huidexcisie neus rechts met een nodulair basaalcelcarcinoom met een invasiediepte van 0,6mm in de centrale lamellen. Na opsnijden puntige uiteinden vrij.

II. Huidexcisie neus rechts met een nodulair basaalcelcarcinoom met een invasiediepte van 1,4mm in de centrale lamellen. Na opsnijden puntige uiteinden vrij.

Voor beide excisies geldt een achtergrond van sebaceuze hyperplasie, passend bij de klinisch aangegeven nevus sebaceus.'''

max_length_sentence = 2048
test_dataset = prepare_test_dataset('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv', tokenizer1, max_length_sentence)
sample = {"Conclusie": input_text, "Codes": ""}
input_ids = tokenizer1.encode(sample["Conclusie"], return_tensors='pt')
model_inputs = preprocess_function(sample, tokenizer1, max_length_sentence)
model_inputs["input_ids"] = input_ids
model_inputs = {k: v.to(model1.device) if v is not None else None for k, v in model_inputs.items()}
output = model1.generate(
    model_inputs["input_ids"],
    max_length=128,
    diversity_penalty=0.3,
    num_beams=6,
    num_beam_groups=2,
)
translated_text = tokenizer1.decode(output[0], skip_special_tokens=True)
pred_words = [get_word_from_code(code) for code in translated_text.split()]
print(f"Input: {input_text}")
print(f"Output: {pred_words}")


# state_dict2 = torch.load(checkpoint_model2, map_location='cpu')  # 'map_location' argument for loading on CPU
# model2.load_state_dict(state_dict2)

# state_dict3 = torch.load(checkpoint_model3, map_location='cpu')  # 'map_location' argument for loading on CPU
# model3.load_state_dict(state_dict3)

# state_dict4 = torch.load(checkpoint_model4, map_location='cpu')  # 'map_location' argument for loading on CPU
# model4.load_state_dict(state_dict4)

# thesaurus_location = '/home/msiepel/snomed_20230426.txt'
# thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

# # Function to get word from code
# def get_word_from_code(code):
#     if code == '[c-sep]' or code == '[C-SEP]':
#         return '[C-SEP]'
#     else:
#         word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
#         return word[0] if len(word) > 0 else 'Unknown'

# for index in range(len(test_dataset_1)):
#     sample_1 = test_dataset_1[index]
#     input_text_1 = sample_1["Conclusie"]
#     gold_standard_1 = sample_1["Codes"]

#     # sample_2 = test_dataset_2[index]
#     # input_text_2 = sample_2["Conclusie"]
#     # gold_standard_2 = sample_2["Codes"]

#     # Generate translation for the input sentence
#     input_ids1 = tokenizer1.encode(input_text_1, return_tensors='pt')
#     # input_ids2 = tokenizer2.encode(input_text_2, return_tensors='pt')
#     # input_ids3 = tokenizer3.encode(input_text_2, return_tensors='pt')
#     # input_ids4 = tokenizer4.encode(input_text_2, return_tensors='pt')

#     output1 = model1.generate(
#                     input_ids1,
#                     max_length=128,
#                     diversity_penalty=0.3,
#                     num_beams=6,
#                     num_beam_groups=2,
#                 )
#     translated_text1 = tokenizer1.decode(output1[0], skip_special_tokens=True)

#     # output2 = model2.generate(input_ids2, max_length=32)
#     # translated_text2 = tokenizer2.decode(output2[0], skip_special_tokens=True)

#     # output3 = model3.generate(input_ids3, max_length=32)
#     # translated_text3 = tokenizer3.decode(output3[0], skip_special_tokens=True)

#     # output4 = model4.generate(input_ids4, max_length=32)
#     # translated_text4 = tokenizer4.decode(output4[0], skip_special_tokens=True)

#     # Print input, output, and gold standard
#     print(f"Input text:        {input_text_1}")
#     print(f"Gold standard:     {gold_standard_1}")
#     print(f"Prediction:        {translated_text1}")
#     label_words = [get_word_from_code(code) for code in gold_standard_1.split()]
#     print("Label Words:     ", ' '.join(label_words))

#     # Convert codes to words for prediction
#     pred_words = [get_word_from_code(code) for code in translated_text1.split()]
    
#     print("Prediction Words:", ' '.join(pred_words))
#     # print(f"Unigram:           {translated_text2}")
#     # print(f"BSP 1:             {translated_text3}")
#     # print(f"BSP 098:           {translated_text4}")
#     print("-"*100)