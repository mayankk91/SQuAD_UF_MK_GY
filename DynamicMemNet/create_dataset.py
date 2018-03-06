import json
from nltk.tokenize import sent_tokenize
import spacy
import os
import zipfile
import tarfile


nlp = spacy.load('en')


def get_glove(glove_zip_file, glove_vectors_file):
    try:
        from urllib.request import urlretrieve, urlopen
    except ImportError:
        from urllib import urlretrieve
        from urllib import urlopen
    # large file - 862 MB
    if (not os.path.isfile(glove_zip_file) and
            not os.path.isfile(glove_vectors_file)):
        urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip",
                    glove_zip_file)
    unzip_single_file(glove_zip_file, glove_vectors_file)

def get_fact_id(map, ans_start):
    for k, v in map.items():
        if v[0] < ans_start < v[1]:
            return k
    return -1

def create_squad_dataset(input_file_name, output_file_name):
    with open(input_file_name, 'r') as file:
        parsed_file = json.load(file)
        data = parsed_file['data']
        len_data = len(data)

        with open(output_file_name, 'w') as writer:
            for i in range(len_data):
                passage = data[i]
                for para in passage['paragraphs']:
                    context = nlp(para['context']).text
                    sentences = sent_tokenize(context)
                    map = {}
                    start, index = 0, 1
                    for sentence in sentences:
                        map[index] = (start, start + len(sentence))
                        start = start + len(sentence)
                        writer.write(str(index) + " " + sentence + "\n")
                        index = index + 1
                    qas = para['qas']
                    for qa in qas:
                        question = nlp(qa['question']).text
                        answer = nlp(qa['answers'][0]['text']).text  # just select best one
                        ans_start = qa['answers'][0]['answer_start']
                        fact_id = get_fact_id(map, ans_start)
                        writer.write(str(index) + " " + question + "\t" + answer + "\t" + str(fact_id) + "\n")
                        index = index + 1

def unzip_single_file(zip_file_name, output_file_name):
    """
        If the output file is already created, don't recreate
        If the output file does not exist, create it from the zipFile
    """
    if not os.path.isfile(output_file_name):
        with open(output_file_name, 'wb') as out_file:
            with zipfile.ZipFile(zip_file_name) as zipped:
                for info in zipped.infolist():
                    if output_file_name in info.filename:
                        with zipped.open(info) as requested_file:
                            out_file.write(requested_file.read())
                            return


def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
    if not os.path.isfile(output_file_name):
        with tarfile.open(zip_file_name) as un_zipped:
            un_zipped.extract(interior_relative_path + output_file_name)

def main():
    train_set_json = "data/train-v1.1.json"
    train_set_file = "data/train-v1.1.txt"
    dev_set_json = "data/dev-v1.1.json"
    dev_set_file = "data/dev-v1.1.txt"
    create_squad_dataset(train_set_json, train_set_file)
    create_squad_dataset(dev_set_json, dev_set_file)

    glove_zip_file = "data/glove.6B.zip"
    glove_vectors_file = "data/glove.6B.50d.txt"
    get_glove(glove_zip_file, glove_vectors_file)



if __name__ == '__main__':
    main()