from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('/home/msiepel/PALGA-Transformers/PALGA-Transformers/T5_small_32128_pretrain_with_codes')

# T	2022	i dunnedarmbiopten zonder afwijkingen ii maagantrum en corpusbiopten met een aspecifiek chronisch en gering actief ontstekingsinfiltraat helicobacter onderzoek is negatief er is enige atrofie		t64000 p11400 m00100 [C-SEP] t63000 p11400 m40000	i dunnedarmbiopten zonder afwijkingen ii maagantrum en corpusbiopten met een aspecifiek chronisch en gering actief ontstekingsinfiltraat helicobacter onderzoek is negatief er is enige atrofie	i dunnedarmbiopten zonder afwijkingen ii maagantrum en corpusbiopten met een aspecifiek chronisch en gering actief ontstekingsinfiltraat helicobacter onderzoek is negatief er is enige atrofie

sent = "i dunnedarmbiopten zonder afwijkingen ii maagantrum en corpusbiopten met een aspecifiek chronisch en gering actief ontstekingsinfiltraat helicobacter onderzoek is negatief er is enige atrofie"
codes = "t64000 p11400 m00100 [C-SEP] t63000 p11400 m40000"


print(tokenizer.tokenize(sent))
print(tokenizer.tokenize(codes))