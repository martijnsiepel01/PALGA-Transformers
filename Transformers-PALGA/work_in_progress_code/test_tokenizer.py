from transformers import AutoTokenizer

# Initialize your tokenizer (replace with your tokenizer)
tokenizer = AutoTokenizer.from_pretrained('/home/msiepel/tokenizer')

def count_tokens(sentence, tokenizer):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    # Return the number of tokens
    return len(tokens)

# Example sentence
sentence = "Input Sequence: I: Hemicolectomie rechts : Er wordt een tumor aangetroffen met beeld van een adenocarcinoom ( 65 %) naast mucineus carcinoom ( 30 %) en zegelringcelcarcinoom ( 5 %). De tumor is matig gedifferentieerd. De maximale grootte bedraagt 6,5 cm. De tumor invadeert tot in het omgevend vetweefsel. Er is een kleine perforatie. Het meest nabij gelegen sneevlak is distaal. Dit sneevlak is vrij. De marge t.o.v. dit sneevlak bedraagt 4 cm. Het circumferentiele snijvlak is vrij. De marge t.o.v. dit sneevlak bedraagt 3 cm. Er is geen vasoinvasieve groei. Geen perineurale invasie. Aantal separate poliepen: 1, waarvan adenomateus laaggradig dysplastisch: 1. Aantal lymfklieren onderzocht: 15. Geen metastasen. Stadieringsvoorstel: pT3, N0 (Stadiering volgens TNM versie 5, conform afspraak werkgroep gastro-intestinale tumoren IKZ regio). II: Low anterior resectie : Status na voorbestraling 5x5 Gy met geringe tumorregressie (Mandard IV). Er wordt een tumor aangetroffen met beeld van een adenocarcinoom. De tumor is matig gedifferentieerd. De maximale grootte bedraagt 3 cm. De tumor invadeert tot in het omgevend vetweefsel. Het meest nabij gelegen sneevlak is distaal. Dit sneevlak is vrij. De marge t.o.v. dit sneevlak bedraagt 2,5 cm. Het circumferentiele snijvlak is vrij. De marge t.o.v. dit sneevlak bedraagt 3 cm. Er is geen vasoinvasieve groei. Geen perineurale invasie. Aantal lymfklieren onderzocht: 21. Geen metastasen. Stadieringsvoorstel: y pT3, N0 (Stadiering volgens TNM versie 5, conform afspraak werkgroep gastro-intestinale tumoren IKZ regio). 4-2-2015: Op verzoek van de oncologievergadering dd 3-2-2015 is MSI-bepaling op beide tumoren ingezet. De uitslag wordt binnen 10 werkdagen verwacht. MOLECULAIRE PATHOLOGIE dd 12-2-2015 MSI-high (in colon rechts) en MSI-stable (in rectum) aangetoond. Dit is informatief in het kader van therapie en niet vanwege een mogelijk associatie met het Lynch syndroom (zie opmerking onder microscopie). AANVULLEND 16-3-2015 (Judith Jeuken): Deze aanvulling is gemaakt vanwege een administratieve wijziging. De medische inhoud blijft ongewijzigd."

# Count the number of tokens in the sentence
num_tokens = count_tokens(sentence, tokenizer)

print("Number of tokens:", num_tokens)
