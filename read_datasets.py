import csv
import numpy as np
import os
import string
import operator
from stemming.porter2 import stem
import matplotlib.pyplot as plt
#from pytagcloud import create_tag_image, make_tags
#from pytagcloud.lang.counter import get_tag_counts
np.set_printoptions(threshold=np.nan)

KC = 10
KD = 10
K = KC + KD

ALPHA = 100
BETA = 10
GAMA = 10

DATASET_DIRECTORY = './datasets/'
IEE_DATASET = "vis_papers.csv"
NEWSGROUPS_DATASET_FOLDER = "20news-bydate-train"
REUTERS_DATASET = "reuters.csv"

PUNCTUATION_TABLE_IN = [p for p in string.punctuation]
PUNCTUATION_TABLE_IN.remove("'")
PUNCTUATION_TABLE_IN = "".join(PUNCTUATION_TABLE_IN)
PUNCTUATION_TABLE_OUT = "".join([" " for _ in PUNCTUATION_TABLE_IN])

STOP_WORDS = [
    "a", "able", "about", "across", "after", "all", "almost", "also", "am",
    "among", "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "but", "by", "can", "cannot", "could", "dear", "did", "do", "does",
    "either", "else", "ever", "every", "for", "from", "get", "got", "had",
    "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i",
    "if", "in", "into", "is", "it", "its", "just", "least", "let", "like",
    "likely", "may", "me", "might", "most", "must", "my", "neither", "no",
    "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our",
    "own", "rather", "said", "say", "says", "she", "should", "since", "so",
    "some", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "yet", "you", "your", ".", " ", "1", "2", "3",
    "4", "5", "6", "7", "8", "9", "0", "during", "changes", "(1)", "(2)",
    "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "usually", "involved",
    "labeled", "10"]

IEE_COLUMNS = {
    'type': 0,
    'year': 1,
    'title': 2,
    'abstract': 10
}


METADATA = 90005
CONTENT = 90006


def read_iee_dataset(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            d = {}
            for key in IEE_COLUMNS:
                if row[IEE_COLUMNS['abstract']].strip() == '':
                    continue
                if key == 'abstract':
                    d[key] = process_content(row[IEE_COLUMNS[key]])
                else:
                    d[key] = row[IEE_COLUMNS[key]]
            data.append(d)
    return data


def fc(w1, w2):
    return np.linalg.norm(w1, w2)


def fd(w1, w2):
    pass


def normalize(a):
    return a / (a.sum(axis=0) + 0.000000000000000001)


def split_iee_by(dataset, split_on):
    result = {}
    for data in dataset:
        if split_on not in data:
            continue

        if data[split_on] not in result:
            result[data[split_on]] = []
        result[data[split_on]].append(data)
    return result


def update_wc(R, W1, W2, X, H, alpha):
    m, k = W1.shape
    m, n1 = X.shape
    ht = H.T.dot(H)
    for l in xrange(KC):
        R[:, l] = (ht[l, l]/(ht[l, l] + n1 * alpha)) * W1[:, l] + (X.dot(H[:, l]) - W1.dot(ht[:, l]) + n1*alpha * W2[:, l])/(ht[l, l] + n1 * alpha + 0.000000000001)
    R[R < 0] = 0

def update_wd(R, W1, W2, X, H, beta):
    m, k = W1.shape
    m, n1 = X.shape
    ht = H.T.dot(H)
    factor = n1 * 0.5 * beta * sum([W2[:, p] for p in xrange(KC, K)])
    for l in xrange(KC, K):
        R[:, l] = W1[:, l] + (X.dot(H[:, l]) - W1.dot(ht[:, l]) - factor) / (ht[l, l] + 0.00000000001)
    R[R < 0] = 0

def update_h(R, W, X, H):
    for l in xrange(K):
        R[:, l] = H[:, l] + (X.T.dot(W)[:, l] - (H.dot(W.T).dot(W)[:, l])) / (W.T.dot(W)[l, l] + 0.0000000000000000001)
    R[R < 0] = 0

def word_table_union(a, b):
    return set(a.keys()).union(b.keys())

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def parse_newsgroup_metadata(metadata):
    pass


def remove_quotations(content):
    """
    Removes all quotations from newsgroup content.
    A quotation is a line that starts with '>'
    """
    return filter(lambda x: not x.startswith('>'), content)


def process_content(content):
    content = remove_quotations(content)
    content = [l.translate(
        string.maketrans(PUNCTUATION_TABLE_IN, PUNCTUATION_TABLE_OUT)
    ) for l in content]
    return ' '.join("".join(content).split()).lower()


def count_words(text, table=None):
    if table is None:
        table = {}
    for dirty_word in text.split():
        word = stem(dirty_word)
        if word in STOP_WORDS:
            continue
        if word not in table:
            table[word] = 1
        else:
            table[word] += 1
    return table


def process_newsgroup_email(file_descriptor):
    f = file_descriptor
    metadata = []
    content = []

    extracting = METADATA

    for line in f.readlines():
        if line == '\n':
            # We assume that at least one \n separates metadata from content
            extracting = CONTENT
            continue
        if extracting == METADATA:
            metadata.append(line.rstrip('\n'))
        else:
            content.append(line.rstrip('\n'))
    return metadata, process_content(content)


def read_newsgroups_dataset(path):
    topics = get_immediate_subdirectories(path)
    print "Found", len(topics), "topics: ", topics
    data = {}

    for topic in topics:
        data[topic] = []
        p = path + '/' + topic
        for filename in (os.listdir(p)):
            with open(p + '/' + filename, 'r') as f:
                metadata, content = process_newsgroup_email(f)
                data[topic].append({
                    "metadata": metadata,
                    "content": content
                })
    return data

def initialize_H(X, W):
    Q, R = np.linalg.qr(W)
    R_inv = np.linalg.inv(R)

    return R_inv.dot(Q.T).dot(X).T

def word_cloud(word_table, outfile='./test_cloud.png'):
    pass


def read_reuters_dataset(path):
    pass


def plot_word_table(word_table, outfile='./test_table.png'):
    word_table = word_table[-5:]
    words = []
    values = []
    for (word, value) in word_table:
        words.append(word)
        values.append(value)

    words = tuple(words)
    print words
    print values
    #words = ("any", "test", "space", "nasa")
    #values = [1, 4, 5, 6]
    plt.barh(values, values, align='center', alpha=0.4)
    plt.yticks(values, words)
    plt.xlabel('Count')
    plt.title('Word count')
    plt.savefig(outfile)



def test_newsgroup_dataset():
    d = read_newsgroups_dataset(DATASET_DIRECTORY + NEWSGROUPS_DATASET_FOLDER)
    table = {}
    for mail in d['sci.space']:
        table = count_words(mail['content'], table)
    table = sorted(table.items(), key=operator.itemgetter(1))
    print len(table)


def test_iee_dataset():
    d = read_iee_dataset(DATASET_DIRECTORY + IEE_DATASET)
    d = split_iee_by(d, 'type')
    vast_table = {}
    for document in d['VAST']:
        vast_table = count_words(document['abstract'], vast_table)
    vast_table_sorted = sorted(vast_table.items(), key=operator.itemgetter(1))
    print "VAST documents:", len(d['VAST'])
    print "VAST keywords - ", len(vast_table_sorted)

    info_table = {}
    for document in d['InfoVis']:
        info_table = count_words(document['abstract'], info_table)
    info_table_sorted = sorted(info_table.items(), key=operator.itemgetter(1))
    print "InfoVis documents:", len(d['InfoVis'])
    print "InfoVis keywords - ", len(info_table_sorted)

    keywords = list(word_table_union(vast_table, info_table))
    print "Total number of keywords:", len(keywords)
    keyword_map = {}
    for i, k in enumerate(keywords):
        keyword_map[k] = i

    vast_table = np.zeros((len(keywords), len(d['VAST'])))
    for i, document in enumerate(d['VAST']):
        word_count = count_words(document['abstract'], {})
        for w in word_count:
            vast_table[keyword_map[w], i] = word_count[w]

    infovis_table = np.zeros((len(keywords), len(d['InfoVis'])))
    for i, document in enumerate(d['InfoVis']):
        word_count = count_words(document['abstract'], {})

        for w in word_count:
            infovis_table[keyword_map[w], i] = word_count[w]
    infovis_table = normalize(infovis_table)

    W1 = normalize(np.random.random((len(keywords), K)))
    W2 = normalize(np.random.random((len(keywords), K)))

    X1 = vast_table
    X2 = infovis_table


    H1 = initialize_H(vast_table, W1)
    H1[H1 < 0] = 0
    H1 = normalize(H1)
    H2 = initialize_H(infovis_table, W2)
    H2[H2 < 0] = 0
    H2 = normalize(H2)

    max_iterations = 20
    for i in xrange(max_iterations):
        #W1_new = np.zeros(W1.shape)
        #W2_new = np.zeros(W2.shape)
        #H1_new = np.zeros(H1.shape)
        #H2_new = np.zeros(H2.shape)

        update_wc(W1, W1, W2, X1, H1, ALPHA)
        update_wd(W1, W1, W2, X1, H1, BETA)

        update_wc(W2, W2, W1, X2, H2, ALPHA)
        update_wd(W2, W2, W1, X2, H2, BETA)

        update_h(H1, W1, X1, H1)
        update_h(H2, W2, X2, H2)

        for l, norm in enumerate(np.linalg.norm(W1, 2, axis=0)):
            H1[:, l] = norm * H1[:, l]
        for l, norm in enumerate(np.linalg.norm(W2, 2, axis=0)):
            H2[:, l] = norm * H2[:, l]

        W1 = normalize(W1)
        W2 = normalize(W2)

    vmap = {}
    for i, value in enumerate(W1[:, 15]):
        vmap[keywords[i]] = value
    ss = sorted(vmap.items(), key=operator.itemgetter(1))[-10:]
    print ss
    print "------"
    print "------"

    vmap = {}
    for i, value in enumerate(W1[:, 16]):
        vmap[keywords[i]] = value
    ss = sorted(vmap.items(), key=operator.itemgetter(1))[-10:]
    print ss
    print "------"
    print "------"

    vmap = {}
    for i, value in enumerate(W1[:, 14]):
        vmap[keywords[i]] = value
    ss = sorted(vmap.items(), key=operator.itemgetter(1))[-10:]
    print ss
    print "------"
    print "------"



if __name__ == "__main__":
    #plot_word_table({})
    test_iee_dataset()
