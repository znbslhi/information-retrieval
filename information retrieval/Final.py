import csv
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Case folding
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

#tokenaise function
def phase1(file_name):
    # Read data from CSV file
    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Preprocess data        
    for i, row in enumerate(data):
        # Preprocess title and plot
        title_tokens = preprocess_text(row['title'])
        plot_tokens = preprocess_text(row['plot'])
        
        # Add id to row
        row['id'] = i+1
        
        # Add preprocessed tokens to row
        row['title_tokens'] = ' '.join(title_tokens)
        row['plot_tokens'] = ' '.join(plot_tokens)
        
        # Add tokens from plot after tokens from title
        row['All_tokens'] = ' '.join(title_tokens) + ' ' + ' '.join(plot_tokens)
    # Write data to CSV file
    with open('train_tokens.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'title_tokens', 'plot_tokens', 'All_tokens'])
        writer.writeheader()
        for row in data:
            writer.writerow({'id': row['id'], 'title_tokens': row['title_tokens'], 'plot_tokens': row['plot_tokens'], 'All_tokens': row['All_tokens']})
#Stop word   
def phase2(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        tokens = []
        for row in reader:
            tokens += row['All_tokens'].split()

    # Count term frequencies
    term_freq = {}
    for token in tokens:
        if token in term_freq:
            term_freq[token] += 1
        else:
            term_freq[token] = 1

    # Sort terms by frequency
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

    # Set the number of stop-words to consider
    num_stopwords = 80

    # Get the most frequent terms as stop-words
    stopwords = [term[0] for term in sorted_terms[:num_stopwords]]

    # Save the stop-words to a file and count term frequency
    with open('stopwords.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Stopword', 'Frequency'])
        for term in sorted_terms[:num_stopwords]:
            writer.writerow([term[0], term[1]])

#positional index
def positional_index():
    train_data = pd.read_csv('train.csv')
    train_tokens = pd.read_csv('train_tokens.csv')

    # Read stop-words from stopwords.csv
    with open('stopwords.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        stopwords = [row['Stopword'] for row in reader]

    # Create positional index for terms
    positional_index = {}
    for i, row in train_tokens.iterrows():
        title_terms = row['title_tokens'].split()
        plot_terms = row['plot_tokens']
        if pd.isna(plot_terms):
            plot_terms = ''
        else:
            plot_terms = plot_terms.split()
        for j, term in enumerate(title_terms):
            if term not in stopwords:
                if term not in positional_index:
                    positional_index[term] = {'title': {}, 'plot': {}}
                if row['id'] not in positional_index[term]['title']:
                    positional_index[term]['title'][row['id']] = []
                positional_index[term]['title'][row['id']].append(j)
        for j, term in enumerate(plot_terms):
            if term not in stopwords:
                if term not in positional_index:
                    positional_index[term] = {'title': {}, 'plot': {}}
                if row['id'] not in positional_index[term]['plot']:
                    positional_index[term]['plot'][row['id']] = []
                positional_index[term]['plot'][row['id']].append(j)

    # Convert positional index to DataFrame
    positional_index_df = pd.DataFrame(positional_index)

    # Save positional index to CSV file
    positional_index_df.to_csv('positional_index.csv', index=False)
    index_file = 'positional_index.csv'
    tokens_file = 'train_tokens.txt'
    stopwords_file = 'Stopwords.txt'
    document_to_add = '1234567,This is the title,This is the plot'
    process_document(document_to_add, index_file, tokens_file, stopwords_file)

#Dynamic indexing
def phase4_drop(id):
    #read train File
    data = pd.read_csv('train.csv')
    data = data.drop(id)
    data.to_csv('train.csv', index=False)
    phase1('train.csv')
    phase2('train_tokens.csv')
    positional_index()

def phase4_insert(insert_string):
    id, plot, title = insert_string.split(",")
    with open('train.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    row_data = [title, plot]
    rows.insert(id, row_data)     
    with open('train.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    phase1('train.csv')
    phase2('train_tokens.csv')
    positional_index()



# Main function
def main():
     phase1('train.csv')
     phase2('train_tokens.csv')
     positional_index()
     phase4_insert()

if __name__ == '__main__':
    main()




