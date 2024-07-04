from transformers import AutoTokenizer


# tokenizer = AutoTokenizer.from_pretrained('/home/gburger01/PALGA-Transformers/PALGA-Transformers/T5_small_32128_pretrain_with_codes')

tokenizer = AutoTokenizer.from_pretrained('google/mT5-small')


print(tokenizer.encode('[C-SEP]', add_special_tokens=False))
print(tokenizer.decode([259, 270, 451, 37210, 421, 1003, 5954, 326, 449, 178928, 331, 1, 0, 0]))