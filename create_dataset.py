import json
from nltk.tokenize import sent_tokenize
import spacy

nlp = spacy.load('en')


def get_fact_id(map, ans_start):
    for k, v in map.items():
        if v[0] < ans_start < v[1]:
            return k
    return -1


with open("dev-v1.1.json", 'r') as file:
    parsed_file = json.load(file)
    data = parsed_file['data']
    len_data = len(data)

    with open("dev.txt", 'w') as writer:
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