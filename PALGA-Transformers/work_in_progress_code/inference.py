from transformers import MT5ForConditionalGeneration, MT5Config, MT5Tokenizer, AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from datasets import load_dataset
import random
import pandas as pd


# Path to your PyTorch checkpoint (.pth) file
checkpoint_model1 = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs15_data_setall_commentall_mt5_small_custom_loss_adafactor_long.pth'
# checkpoint_model2 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentunigram_custom_tokenizer_vocab_16000_1_patience3_freezeallbutxlayers0.pth'
# checkpoint_model3 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentcustom_tokenizer_vocab_16000_patience3_freezeallbutxlayers0.pth'
# checkpoint_model4 = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/models/trained_models/epochs15_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentcustom_tokenizer_vocab_16000_098_patience3_freezeallbutxlayers0.pth'

# # Define the configuration for your MT5 model
config = T5Config.from_pretrained('/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/mT5-small/config.json')

tokenizer1 = MT5Tokenizer.from_pretrained('google/mT5-small')
# tokenizer2 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/unigram_t5_custom_16000_1.model')
# tokenizer3 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/t5_custom_16000_1.model')
# tokenizer4 = T5Tokenizer('/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_16000/t5_custom_16000_098.model')


# # config = MT5Config.from_pretrained("google/flan-t5-small")
# # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

max_length_sentence = 2048

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex for ex in examples["Conclusie"]]
    targets = [ex for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
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


test_dataset_1 = prepare_test_dataset('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv', tokenizer1, max_length_sentence)
# test_dataset_2 = prepare_test_dataset('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_P1.tsv', tokenizer2, max_length_sentence)

model1 = T5ForConditionalGeneration(config)
model1.resize_token_embeddings(len(tokenizer1))

# model2 = T5ForConditionalGeneration(config)
# model2.resize_token_embeddings(len(tokenizer2))

# model3 = T5ForConditionalGeneration(config)
# model3.resize_token_embeddings(len(tokenizer3))

# model4 = T5ForConditionalGeneration(config)
# model4.resize_token_embeddings(len(tokenizer4))

state_dict1 = torch.load(checkpoint_model1, map_location='cpu')  # 'map_location' argument for loading on CPU
model1.load_state_dict(state_dict1)

# state_dict2 = torch.load(checkpoint_model2, map_location='cpu')  # 'map_location' argument for loading on CPU
# model2.load_state_dict(state_dict2)

# state_dict3 = torch.load(checkpoint_model3, map_location='cpu')  # 'map_location' argument for loading on CPU
# model3.load_state_dict(state_dict3)

# state_dict4 = torch.load(checkpoint_model4, map_location='cpu')  # 'map_location' argument for loading on CPU
# model4.load_state_dict(state_dict4)

thesaurus_location = '/home/msiepel/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

# Function to get word from code
def get_word_from_code(code):
    if code == '[c-sep]' or code == '[C-SEP]':
        return '[C-SEP]'
    else:
        word = thesaurus[(thesaurus['DEPALCE'].str.lower() == code.lower()) & (thesaurus['DESTACE'] == 'V')]['DETEROM'].values
        return word[0] if len(word) > 0 else 'Unknown'

for index in range(len(test_dataset_1)):
    sample_1 = test_dataset_1[index]
    input_text_1 = sample_1["Conclusie"]
    gold_standard_1 = sample_1["Codes"]

    # sample_2 = test_dataset_2[index]
    # input_text_2 = sample_2["Conclusie"]
    # gold_standard_2 = sample_2["Codes"]

    # Generate translation for the input sentence
    input_ids1 = tokenizer1.encode(input_text_1, return_tensors='pt')
    # input_ids2 = tokenizer2.encode(input_text_2, return_tensors='pt')
    # input_ids3 = tokenizer3.encode(input_text_2, return_tensors='pt')
    # input_ids4 = tokenizer4.encode(input_text_2, return_tensors='pt')

    output1 = model1.generate(
                    input_ids1,
                    max_length=128,
                    diversity_penalty=0.3,
                    num_beams=6,
                    num_beam_groups=2,
                )
    translated_text1 = tokenizer1.decode(output1[0], skip_special_tokens=True)

    # output2 = model2.generate(input_ids2, max_length=32)
    # translated_text2 = tokenizer2.decode(output2[0], skip_special_tokens=True)

    # output3 = model3.generate(input_ids3, max_length=32)
    # translated_text3 = tokenizer3.decode(output3[0], skip_special_tokens=True)

    # output4 = model4.generate(input_ids4, max_length=32)
    # translated_text4 = tokenizer4.decode(output4[0], skip_special_tokens=True)

    # Print input, output, and gold standard
    print(f"Input text:        {input_text_1}")
    print(f"Gold standard:     {gold_standard_1}")
    print(f"Prediction:        {translated_text1}")
    label_words = [get_word_from_code(code) for code in gold_standard_1.split()]
    print("Label Words:     ", ' '.join(label_words))

    # Convert codes to words for prediction
    pred_words = [get_word_from_code(code) for code in translated_text1.split()]
    
    print("Prediction Words:", ' '.join(pred_words))
    # print(f"Unigram:           {translated_text2}")
    # print(f"BSP 1:             {translated_text3}")
    # print(f"BSP 098:           {translated_text4}")
    print("-"*100)