import csv
import pandas as pd
import editdistance

# خواندن فایل positional_index.csv و استخراج توکن ها
df = pd.read_csv('positional_index.csv', nrows=1)
tokens = df.columns.tolist()

# ساخت دیکشنری با استفاده از توکن ها
dictionary = {i: str(token) for i, token in enumerate(tokens)}

def get_closest_words(query, dictionary, output_file):
    query_bigrams = set([query[i:i+2] for i in range(len(query)-1)])
    
    closest_words = []
    max_jaccard = 0
    closest_word = ''
    
    for word in dictionary:
        word_bigrams = set([dictionary[word][i:i+2] for i in range(len(dictionary[word])-1)])
        jaccard = len(query_bigrams.intersection(word_bigrams)) / len(query_bigrams.union(word_bigrams))
        
        if len(closest_words) < 10:
            closest_words.append((dictionary[word], jaccard))
            closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)
            if jaccard > max_jaccard:
                closest_word = dictionary[word]
                max_jaccard = jaccard
        elif jaccard > closest_words[-1][1]:
            closest_words[-1] = (dictionary[word], jaccard)
            closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)
            if jaccard > max_jaccard:
                closest_word = dictionary[word]
                max_jaccard = jaccard
    
    closest_words_with_edit_distance = [(word, editdistance.eval(query, word), jaccard) for word, jaccard in closest_words]
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Edit Distance', 'Jaccard Similarity'])
        for row in closest_words_with_edit_distance:
            writer.writerow(row)
    
    return closest_word

query = "grompier"
output_file = 'closest_words.csv'
closest_word = get_closest_words(query, dictionary, output_file)
print(f"The closest word to '{query}' is '{closest_word}'")

