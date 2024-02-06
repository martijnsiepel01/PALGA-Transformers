from transformers import MT5ForConditionalGeneration, MT5Config, MT5Tokenizer, AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from datasets import load_dataset
import random

# Path to your PyTorch checkpoint (.pth) file
checkpoint_model1 = '/home/msiepel/models/trained_models/epochs50_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelmT5_small_commentmT5_small_baseline_patience5.pth'
checkpoint_model2 = '/home/msiepel/models/trained_models/epochs50_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelmT5_base_commentmT5_base_baseline_patience5.pth'
checkpoint_model3 = '/home/msiepel/models/trained_models/epochs50_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-small_commentflan-t5-small_baseline_patience5.pth'
checkpoint_model4 = '/home/msiepel/models/trained_models/epochs50_maxlengthsentence512_trainbatchsize8_validationbatchsize4_lr0.0001_maxgeneratelength32_datasetall_modelflan-t5-base_commentflan-t5-base_baseline_patience5_freezeallbutxlayers0.pth'

# # Define the configuration for your MT5 model
config1 = MT5Config.from_pretrained('/home/msiepel/models/mT5_small/config.json')
config2 = MT5Config.from_pretrained('/home/msiepel/models/mT5_base/config.json')
config3 = T5Config.from_pretrained('/home/msiepel/models/flan-t5-small/config.json')
config4 = T5Config.from_pretrained('/home/msiepel/models/flan-t5-base/config.json')


tokenizer1 = MT5Tokenizer.from_pretrained('/home/msiepel/tokenizer')
tokenizer2 = T5Tokenizer.from_pretrained('/home/msiepel/flan_tokenizer')

# # config = MT5Config.from_pretrained("google/flan-t5-small")
# # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

max_length_sentence = 512

def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex for ex in examples["Conclusie"]]
    targets = [ex for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_test_dataset(tokenizer, max_length_sentence):
    # test_data_location = "/home/msiepel/data/all/all_norm_test.tsv"
    test_data_location = "/home/msiepel/data/gold_P1.tsv"
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


test_dataset = prepare_test_dataset(tokenizer1, max_length_sentence)


model1 = MT5ForConditionalGeneration(config1)
model1.resize_token_embeddings(len(tokenizer1))

model2 = MT5ForConditionalGeneration(config2)
model2.resize_token_embeddings(len(tokenizer1))

model3 = T5ForConditionalGeneration(config3)
model3.resize_token_embeddings(len(tokenizer2))

model4 = T5ForConditionalGeneration(config4)
model4.resize_token_embeddings(len(tokenizer2))

state_dict1 = torch.load(checkpoint_model1, map_location='cpu')  # 'map_location' argument for loading on CPU
model1.load_state_dict(state_dict1)

state_dict2 = torch.load(checkpoint_model2, map_location='cpu')  # 'map_location' argument for loading on CPU
model2.load_state_dict(state_dict2)

state_dict3 = torch.load(checkpoint_model3, map_location='cpu')  # 'map_location' argument for loading on CPU
model3.load_state_dict(state_dict3)

state_dict4 = torch.load(checkpoint_model4, map_location='cpu')  # 'map_location' argument for loading on CPU
model4.load_state_dict(state_dict4)


# Get 100 random indices from the training dataset
random_indices = random.sample(range(len(test_dataset)), 100)
for index in random_indices:
    sample = test_dataset[index]
    input_text = sample["Conclusie"]
    gold_standard = sample["Codes"]

    # Generate translation for the input sentence
    input_ids1 = tokenizer1.encode(input_text, return_tensors='pt')
    input_ids2 = tokenizer2.encode(input_text, return_tensors='pt')

    output1 = model1.generate(input_ids1, max_length=32)
    translated_text1 = tokenizer1.decode(output1[0], skip_special_tokens=True)

    output2 = model2.generate(input_ids1, max_length=32)
    translated_text2 = tokenizer1.decode(output1[0], skip_special_tokens=True)

    output3 = model3.generate(input_ids2, max_length=32)
    translated_text3 = tokenizer2.decode(output3[0], skip_special_tokens=True)

    output4 = model4.generate(input_ids2, max_length=32)
    translated_text4 = tokenizer2.decode(output4[0], skip_special_tokens=True)

    # Print input, output, and gold standard
    print(f"Input text: {input_text}")
    print(f"Gold standard: {gold_standard}")
    print(f"mT5-small: {translated_text1}")
    print(f"mT5-base: {translated_text2}")
    print(f"flan-t5-small: {translated_text3}")
    print(f"flan-t5-base: {translated_text4}")
    print("-"*100)