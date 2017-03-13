vocab_file = "../../data/vocab_all.txt"
# glove_file = "../../data/glove.6B/glove.6B.100d.txt"
# trimmed_glove_file = "../../data/glove.6B/glove.6B.100d_trimmed.txt"
glove_file = "../../data/glove.840B.300d/glove.840B.300d.txt"
trimmed_glove_file = "../../data/glove.840B.300d/glove.840B.300d_trimmed.txt"

all_words = set()
with open(vocab_file) as v_in:
	for line in v_in:
		word = line.split()[0]
		word = word.lstrip("-")
		all_words.add(word.lower())
print len(all_words)

with open(glove_file) as ifs:
	with open(trimmed_glove_file, 'wr') as outputfs:	
	    for line in ifs:
	        line = line.strip()
	        if not line:
	            continue
	        row = line.split()
	        token = row[0].lower()
	        if token in all_words:
	        	outputfs.write(line + "\n")
	        	all_words.remove(token)

print len(all_words)
missing_words = set()
count = 0
for word in all_words:
	if "-" in word:
		count += 1
	else:
		missing_words.add(word)
print count
print missing_words

def average_neighbors(word_vec_matrix, vocab, missing_indices, window_size=4):
	missing_words = [vocab.token_for_id(i) for i in missing_indices]
	neighbors_count = {}
	for missing_word in missing_words:
		neighbors_count[missing_word] = 0

    for dataset in ["snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl"]:
        for line in open(os.path.join(dataset_path, dataset), "r").readlines():
            data = json.loads(line)
            sentence = word_tokenize(data["sentence1"].lower())
        	for missing_word in missing_words:
        		if missing_word in sentence:
        			missing_sentence_index = sentence.find(missing_word)

        			for neighbor_index in xrange(max(missing_sentence_index - window_size, 0), 
									min(len(sentence_words), missing_sentence_index + window_size))
						if neighbor_index != missing_sentence_index and 
								sentence[neighbor_index] not in missing_words:
							missing_index = vocab.id_for_token(missing_word)	
					        matrix[missing_index] += matrix[vocab.id_for_token(sentence[neighbor_index])]
    						neighbors_count[missing_word] += 1

    for missing_word in missing_words:
    	if neighbors_count[missing_word] > 0:
    		index = vocab.id_for_token(token)
    		matrix[index] /= neighbors_count[missing_word]    

    return word_vec_matrix