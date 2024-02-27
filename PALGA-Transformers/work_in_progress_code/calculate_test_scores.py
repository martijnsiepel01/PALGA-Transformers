from transformers import T5Tokenizer, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm


def preprocess_function(examples, tokenizer, max_length_sentence):
    inputs = [ex.lower() for ex in examples["Conclusie"]]
    targets = [ex.lower() for ex in examples["Codes"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length_sentence, truncation=True
    )
    return model_inputs

def prepare_test_dataset(test_data_location, tokenizer, max_length_sentence, report_type=None, min_conclusion_length=None, max_conclusion_length=None):
    # Load and initially filter the dataset
    dataset = load_dataset("csv", data_files=test_data_location, delimiter="\t")
    dataset = dataset.filter(lambda example: example["Codes"] is not None and example["Codes"] != '')
    
    # Filter by report type if specified
    if report_type is not None:
        dataset = dataset.filter(lambda example: example["Type"] == report_type)
    
    # Filter by conclusion length if specified
    if min_conclusion_length is not None or max_conclusion_length is not None:
        def conclusion_length_filter(example):
            conclusion_length = len(example["Conclusie"])
            if min_conclusion_length is not None and conclusion_length < min_conclusion_length:
                return False
            if max_conclusion_length is not None and conclusion_length > max_conclusion_length:
                return False
            return True
        dataset = dataset.filter(conclusion_length_filter)
    
    # Tokenize and preprocess the dataset
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length_sentence),
        batched=True
    )
    
    # Remove unnecessary columns and set format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["Conclusie", "Codes"])
    tokenized_datasets.set_format("torch")
    
    # Select the subset of the dataset
    test_dataset = tokenized_datasets['train']
    return test_dataset

def test_step(model, dataloader, tokenizer, max_generate_length):
    metric = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')

    model.eval()
    total_test_loss = 0.0
    test_preds = []
    test_labels = []
    decoded_test_preds = []  
    decoded_test_inputs = []  # Added line

    for batch in tqdm(dataloader, desc="Testing"):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_generate_length,
            )

        labels = batch["labels"]
        loss = model(**batch).loss
        total_test_loss += loss.item()

        filtered_generated_tokens = [token[token != -100] for token in generated_tokens]
        filtered_labels = [label[label != -100] for label in labels]

        test_preds.extend(filtered_generated_tokens)
        test_labels.extend(filtered_labels)

        # Decode original input sequence
        decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        decoded_test_inputs.extend(decoded_inputs)

    decoded_test_preds = tokenizer.batch_decode(test_preds, skip_special_tokens=True)
    decoded_test_labels = tokenizer.batch_decode(test_labels, skip_special_tokens=True)
    metric.add_batch(predictions=decoded_test_preds, references=decoded_test_labels)
    rouge.add_batch(predictions=decoded_test_preds, references=decoded_test_labels)

    # Compute BLEU score
    bleu_score = metric.compute()['score']
    # Compute ROUGE scores
    rouge_score = rouge.compute()
    ROUGE_1 = rouge_score['rouge1'] 
    ROUGE_2 = rouge_score['rouge2'] 
    ROUGE_L = rouge_score['rougeL'] 
    ROUGE_Lsum = rouge_score['rougeLsum'] 
    bleu_score = bleu_score / 100

    average_rouge_test = (ROUGE_1 + ROUGE_2 + ROUGE_L + ROUGE_Lsum)/4
    
    test_metrics = {
        "bleu": bleu_score,
        "average rouge": average_rouge_test,
    }
    
    return test_metrics, decoded_test_preds, decoded_test_labels, decoded_test_inputs


def prepare_datacollator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator

def prepare_dataloader(dataset, data_collator, batch_size):
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )
    return dataloader

tokenizer_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/flan_tokenizer"
# config= "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/flan-t5-small/config.json"
# checkpoint = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/models/trained_models/num_train_epochs40_data_sethisto_commentpredict_codes_test_everything_default.pth"
tokenizer = T5Tokenizer(tokenizer_path)
# model = T5ForConditionalGeneration(config)
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load(checkpoint))
# collator = prepare_datacollator(tokenizer, model)
test_dataset_path = "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test_with_codes.tsv"
test_dataset = prepare_test_dataset(test_data_location=test_dataset_path, tokenizer=tokenizer, max_length_sentence=512, report_type="T", min_conclusion_length=None, max_conclusion_length=None)

print(test_dataset)