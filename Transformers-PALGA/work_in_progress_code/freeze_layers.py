from transformers import MT5Tokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AdamW

local_model_path = "/home/msiepel/models/mT5_small"
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path, local_files_only=True)


# Freeze all the parameters in the model first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two encoder layers
for param in model.encoder.block[-2:].parameters():
    param.requires_grad = True

# Unfreeze the last two decoder layers
for param in model.decoder.block[-2:].parameters():
    param.requires_grad = True

# Optionally, print out which layers are trainable to verify
for name, param in model.named_parameters():
    print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")
