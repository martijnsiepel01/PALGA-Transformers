import nltk		# download 'punkt'
import re	
import clr		# install pythonnet	
import sys
from pathlib import Path
import pandas as pd

from pythonnet import load
load("mono")

# nltk.download('punkt')
clr.AddReference (fr"{Path.cwd()}\vertaalmodule.dll")
from vertaalmodule import PalgaCodeCompiler

# -- class containing a PALGA term and code and their characteristics:
class palga_term:
	def __init__(self, term:str, code:str) -> None:
		if ' ' in term:
			self.term = f"'{term}'"
		else:
			self.term = term
		self.code = code

	def __str__(self) -> str:
		return f'{self.term}({self.code})'
	
	def __repr__(self) -> str:
		return f'{self.term}({self.code})'
	
	def is_topography(self) -> bool:
		return self.code.startswith('T') and not self.code.startswith('TYY9')	# TYY9 is laterality
		
	def is_laterality(self) -> bool:
		return self.code.startswith('TYY9')

	def is_diagnosis(self) -> bool:
		return self.code[0] in 'MDEF'
	
	def is_denial(self) -> bool:
		return self.code.startswith('Q')
	
pcc = PalgaCodeCompiler()

# -- decodes the vertaalmodule output
termcode_regex = re.compile(r"(\w+([ .'/+(),-]+\w+)*)[.)-]?\((\w+)\)")
# -- 2 regexes to recognize the sentence where the autopsy information starts in the conclusion text
obductie_regex = re.compile(r"[Bb](ij|lijkens)\s+(\w+\s+)?(lichaams|hersen)?(obductie|sectie|schouwing)")
obductie_regex2 = re.compile(r"[Oo]bductie\s+(toon|laa)t")
# -- leave these codes out: 
codes_not_used = ['__DONT_CODE__', 'F96210', 'F96220']	# don't code this sentence code, man, woman

# -- gets a valid PALGA term instance from vertaalmodule output:
def get_term(term_plus_code: str) -> palga_term:
	m = termcode_regex.match(term_plus_code)
	if not m:
		return None
	
	if m.group(3) in codes_not_used:
		return None
	
	return palga_term(str(m.group(1)), str(m.group(3)))

# -- returns a list of valid PALGA code instances from the output of the vertaalmodule (in 'research' modus)
def get_terms(pcc_output:str) -> list:
	termcodes = pcc_output.split('|')
	terms = []

	for termcode in termcodes:
		term = get_term(termcode)
		if term:
			terms.append(term)
	
	return terms
	
def remove_last_csep(sentence):
    last_index = sentence.rfind("[C-SEP]")
    if last_index != -1:
        return sentence[:last_index]
    else:
        return sentence
	
def compile(conc: str) -> str:
	return pcc.Compile(conc, 'annotate.py')

# -- main method:
# -- split the autopsy conclusion text into sentences using nltk
# -- extract a list of PALGA terms from them in reverse order and only add them to the output if the list has a topography, a diagnosis and NOT a denial
# -- stop after the autopsy regex has been detected (if present), then reverse again to restore the original order
# -- outputs the sentences followed by the detected codes.
def split_and_compile(conc:str):
	sentences = nltk.sent_tokenize(conc, language="dutch")
	terms = []
	i = len(sentences) - 1
	sentencecodes = []
	complete_sentence = ""
	complete_sentence_CSEP = ""
	while i >= 0:
		codes = compile(sentences[i])
		sentencecodes = get_terms(codes)
		# -- only add the sentence if it has a topography, a diagnosis and NOT a denial:
		if has_topo(sentencecodes) and has_diagnosis(sentencecodes) and not has_denial(sentencecodes):
			complete_sentence = sentences[i] + " " + complete_sentence
			complete_sentence_CSEP = sentences[i] + " [C-SEP] " + complete_sentence_CSEP
			sentencecodestring = ''
			for sc in sentencecodes:
				sentencecodestring += str(sc)
				sentencecodestring += ' '
			sentencecodestring = sentencecodestring.strip()
			terms.append(f'{sentences[i]}\n{sentencecodestring}')
		if obductie_regex.match(sentences[i]) or obductie_regex2.match(sentences[i]):
			break
		i -= 1

	terms.reverse()
	complete_sentence_CSEP = remove_last_csep(complete_sentence_CSEP)
	return complete_sentence, complete_sentence_CSEP
	# return terms


def validate_conc(conc:str):
	codes = compile(conc)
	sentencecodes = get_terms(codes)
	if has_topo(sentencecodes) and has_diagnosis(sentencecodes):
		return True
	return False


# -- does the list of PALGA terms contain a diagnosis?
def has_diagnosis(palga_term_list: list) -> bool:
	for term in palga_term_list:
		if term.is_diagnosis():
			return True
	
	return False


# -- does the list of PALGA terms contain a topography?
def has_topo(palga_term_list: list) -> bool:
	for term in palga_term_list:
		if term.is_topography():
			return True
	
	return False

# -- does the list of PALGA terms contain a denial?
def has_denial(palga_term_list: list) -> bool:
	for term in palga_term_list:
		if term.is_denial():
			return True
	
	return False


input_location = "C:/Users/Martijn/OneDrive/Thesis/data/autopsies/autopsies_norm_train_with_codes.tsv"
output_location_1 = "C:/Users/Martijn/OneDrive/Thesis/data/autopsies/autopsies_norm_train_with_codes_split.tsv"
output_location_2 = "C:/Users/Martijn/OneDrive/Thesis/data/autopsies/autopsies_norm_train_with_codes_split_w_CSEP.tsv"

with open(input_location, 'r') as fin, \
     open(output_location_1, 'w') as fout1, \
     open(output_location_2, 'w') as fout2:
		for line in fin:
			line = line.strip()
			cols = line.split('\t')
			if len(cols) >= 4:  # Check if there are at least 2 columns
				coded_sentences, coded_sentences_CSEP = split_and_compile(cols[0])  # Assuming you have the split_and_compile function
				if len(coded_sentences) > 0:
					fout1.write(coded_sentences + '\t' + cols[3] + '\n')  # Write coded sentences and original Codes to file 1
				if len(coded_sentences_CSEP) > 0:
					fout2.write(coded_sentences_CSEP + '\t' + cols[3] + '\n')  # Write coded sentences with CSEP and original Codes to file 2

