import pandas as pd
from transformers import AutoTokenizer


def load_thesaurus(thesaurus_location):
    thesaurus = pd.read_csv(thesaurus_location, sep="|", encoding='latin')
    print(thesaurus.head())
    return thesaurus


def get_unique_codes(thesaurus):
    unique_codes = thesaurus[thesaurus["DESTACE"] == "V"]["DEPALCE"].str.lower().unique().tolist()
    print("Unique codes read successfully")
    return unique_codes


def divide_codes(unique_codes):
    topography = [code for code in unique_codes if code.startswith("t")]
    procedure = [code for code in unique_codes if code.startswith("p")]
    morphology = [code for code in unique_codes if not code.startswith("t") and not code.startswith("p")]
    return topography, procedure, morphology


def tokenize_codes(tokenizer, codes):
    tokens_set = set()
    for word in codes:
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_set.update(token_ids)
    return tokens_set


def create_tokenized_sets(tokenizer, topography, procedure, morphology):
    topography_tokens = tokenize_codes(tokenizer, topography)
    procedure_tokens = tokenize_codes(tokenizer, procedure)
    morphology_tokens = tokenize_codes(tokenizer, morphology)
    print("Topograph, procedure, and morphology words loaded successfully")
    return topography_tokens, procedure_tokens, morphology_tokens


def load_data(data_location):
    # data = pd.read_csv(data_location, sep="\t", usecols=["Codes"], nrows=100000)
    data = pd.read_csv(data_location, sep="\t", usecols=["Codes"])
    # Convert to lower case and split
    data['Codes'] = data['Codes'].str.lower().str.split()
    # Replace '[c-sep]' with '[C-SEP]'
    data['Codes'] = data['Codes'].apply(lambda codes: [code.replace('[c-sep]', '[C-SEP]') for code in codes])
    data.dropna(inplace=True)
    
    # Find rows containing '[C-SEP]'
    contains_c_sep = data[data['Codes'].apply(lambda codes: '[C-SEP]' in codes)]
    
    print(data.head())
    print("First 5 rows containing '[C-SEP]':")
    print(contains_c_sep.head(5))
    
    return data


def split_on_token(lst, token='[c-sep]'):
    parts = []
    start = 0
    while token in lst[start:]:
        index = lst[start:].index(token) + start
        parts.append(lst[start:index])
        start = index + 1
    if start < len(lst):
        parts.append(lst[start:])
    return parts


def split_data_on_token(data, token='[c-sep]'):
    data['Codes'] = data['Codes'].apply(lambda x: split_on_token(x, token))
    print(data.head())
    return data


def explode_data(data):
    data_exploded = data.explode('Codes').reset_index(drop=True)
    print(data_exploded.head())
    return data_exploded


def tokenize_exploded_data(data_exploded, tokenizer):
    data_exploded['Encoded_Codes'] = data_exploded['Codes'].apply(
        lambda x: tokenizer.encode(' '.join(x), add_special_tokens=False) if isinstance(x, list) else []
    )
    print(data_exploded.head())
    return data_exploded


def load_mutually_exclusive_terms(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().lower().split(',') for line in file]
    mutually_exclusive_terms = pd.DataFrame(data, columns=['Term1', 'Term2'])
    return mutually_exclusive_terms


def create_exclusive_dict(mutually_exclusive_terms):
    exclusive_dict = {}
    for index, row in mutually_exclusive_terms.iterrows():
        term1 = int(row['Term1'])
        term2 = int(row['Term2'])
        exclusive_dict[term1] = term2
        exclusive_dict[term2] = term1
    return exclusive_dict


class Trie:
    def __init__(self, base_dict=None, tokenizer=None, exclusive_dict=None, topography_tokens=None, procedure_tokens=None, morphology_tokens=None):
        self.trie_dict = base_dict if base_dict is not None else {}
        self.tokenizer = tokenizer
        self.c_sep_tokens = self.tokenize_c_sep() if tokenizer else None
        self.node_metadata = {}
        self.exclusive_dict = exclusive_dict if exclusive_dict is not None else {}
        self.topography_tokens = topography_tokens
        self.procedure_tokens = procedure_tokens
        self.morphology_tokens = morphology_tokens

    def tokenize_c_sep(self):
        return self.tokenizer.encode("[C-SEP]", add_special_tokens=False) if self.tokenizer else []

    def add(self, sequence):
        current_dict = self.trie_dict
        current_type = "topography"

        for i, code in enumerate(sequence):
            if code not in current_dict:
                current_dict[code] = {}

            if not self.is_valid_transition(current_type, code):
                continue

            current_dict = current_dict[code]

            if code in self.topography_tokens:
                self._add_tokenized_c_sep(current_dict)
                current_type = "topography"
            elif code in self.procedure_tokens:
                current_type = "procedure"
            elif code in self.morphology_tokens:
                self._add_tokenized_c_sep(current_dict)
                current_type = "morphology"

            node_key = self._get_node_key(current_dict)
            self.node_metadata[node_key] = {'type': current_type}

    def is_valid_transition(self, current_type, code):
        if current_type == "topography":
            return code in self.topography_tokens or code in self.procedure_tokens or code in self.morphology_tokens
        elif current_type == "procedure":
            return code in self.procedure_tokens or code in self.morphology_tokens
        elif current_type == "morphology":
            return code in self.morphology_tokens or code in self.topography_tokens
        return False

    def _add_tokenized_c_sep(self, current_dict):
        if self.c_sep_tokens is None:
            raise ValueError("C-SEP tokens are not set")
        for token in self.c_sep_tokens:
            if token not in current_dict:
                current_dict[token] = {}
            current_dict = current_dict[token]

    def get(self, prefix_sequence):
        current_dict = self.trie_dict
        for code in prefix_sequence:
            if code in current_dict:
                current_dict = current_dict[code]
            else:
                return []

        possible_continuations = list(current_dict.keys())

        for code in prefix_sequence:
            if code in self.exclusive_dict:
                mutually_exclusive = self.exclusive_dict[code]
                possible_continuations = [code for code in possible_continuations if code != mutually_exclusive]

        possible_continuations = [code for code in possible_continuations if code not in prefix_sequence]

        node_key = self._get_node_key(current_dict)
        if node_key in self.node_metadata and self.node_metadata[node_key].get('type') == 'morphology':
            possible_continuations.append(self.tokenizer.eos_token_id)

        return possible_continuations

    def _get_node_key(self, node):
        return id(node)

    def get_statistics(self):
        stats = {}
        def traverse(node, depth=0):
            if depth not in stats:
                stats[depth] = 0
            stats[depth] += 1
            for child in node:
                traverse(node[child], depth + 1)
        traverse(self.trie_dict)
        return stats

    def count_unique_codes(self, unique_codes):
        counts = {code: 0 for code in unique_codes}
        def traverse(node):
            for key in node:
                if key in counts:
                    counts[key] += 1
                traverse(node[key])
        traverse(self.trie_dict)
        return counts

    def __repr__(self):
        return self._str_helper(self.trie_dict, "", "Root")

    def __str__(self):
        return self.__repr__()

    def _str_helper(self, current_dict, indent, current_code):
        if not current_dict:
            return f"{indent}{current_code}\n"
        result = f"{indent}{current_code}\n"
        indent += "    "
        for code in current_dict:
            result += self._str_helper(current_dict[code], indent, code)
        return result


def create_palga_trie(thesaurus_location, tokenizer_location, data_location, exclusive_terms_file_path):
    # Load thesaurus and tokenizer
    thesaurus = load_thesaurus(thesaurus_location)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)

    # Get unique codes and divide them
    unique_codes = get_unique_codes(thesaurus)
    topography, procedure, morphology = divide_codes(unique_codes)

    # Create tokenized sets
    topography_tokens, procedure_tokens, morphology_tokens = create_tokenized_sets(tokenizer, topography, procedure, morphology)

    # Load and process data
    data = load_data(data_location)
    data = split_data_on_token(data)
    data_exploded = explode_data(data)
    data_exploded = tokenize_exploded_data(data_exploded, tokenizer)

    # Load mutually exclusive terms
    mutually_exclusive_terms = load_mutually_exclusive_terms(exclusive_terms_file_path)
    exclusive_dict = create_exclusive_dict(mutually_exclusive_terms)

    # Create and populate the Trie
    Palga_trie = Trie(tokenizer=tokenizer, exclusive_dict=exclusive_dict, topography_tokens=topography_tokens, procedure_tokens=procedure_tokens, morphology_tokens=morphology_tokens)
    for index, row in data_exploded.iterrows():
        codes = row['Encoded_Codes']
        Palga_trie.add(codes)

    return Palga_trie

