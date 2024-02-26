from transformers import T5Tokenizer

# Initialize the FLAN-T5 tokenizer
flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Assuming your custom tokenizer has a similar interface (you will need to adjust this part)
# For demonstration, replace `your_custom_tokenizer_module.YourCustomTokenizerClass` with your actual class
custom_tokenizer = T5Tokenizer.from_pretrained("/home/gburger01/PALGA-Transformers/PALGA-Transformers/custom_t5_tokenizer_32128/combined_data_unigram_t5_custom_32128_1_lower_case.model")

def compare_tokenization(input_sentence):
    # Tokenize with FLAN-T5 tokenizer
    flan_t5_tokens = flan_t5_tokenizer.tokenize(input_sentence)
    print("FLAN-T5 Tokens:", flan_t5_tokens)
    
    # Tokenize with your custom tokenizer
    # You will need to replace `tokenize` with the actual method you use to tokenize the sentence
    custom_tokens = custom_tokenizer.tokenize(input_sentence)
    print("Custom Tokenizer Tokens:", custom_tokens)

# Example usage
input_sentence = "Punctie lymfklier lies links: Lokalisatie/metastase plaveiselcelcarcinoom. Het materiaal is suboptimaal voor immunohistochemie, derhalve kan er geen uitspraak worden gedaan of dit een HPV-geassocieerde laesie betreft. Zie ook H23-13525.".lower()
compare_tokenization(input_sentence)
