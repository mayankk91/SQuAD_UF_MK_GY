import json
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from ngram import NGram
import re

class DataAugmentor:
    def write_context_sentences_to_file(self):
        filename = "context_sentences_orig_correct.txt"
        file = open(filename, "wb")
        dataset = json.load(open("/home/mayank/data/squad/train-v1.1.json"))
        data = dataset["data"]
        for ele in data:
            title = ele["title"]
            for paragraph in ele["paragraphs"]:
                context = paragraph["context"]
                context_sentences = sent_tokenize(context)
                context_sentences = '\n'.join(context_sentences).encode('utf-8')
                file.write(context_sentences)
                file.write("\n".encode('utf-8'))
        file.close()

    def create_augmented_data(self):
        file_orig = open("context_sentences_orig_correct.txt", "rb")
        file_translated = open("context_sentences_translated_copy_fix.txt", "rb")
        # k = 5
        k = 1
        dict = {}
        for orig_sentence in file_orig.readlines():
            idx = 0
            translated_options = []
            while idx < (k*k):
                translated_options.append(file_translated.readline().decode('utf-8').strip())
                idx += 1
            dict[orig_sentence.decode('utf-8').strip()] = translated_options
        file_orig.close()
        file_translated.close()


        dataset = json.load(open("train-v1.1.json"))
        data = dataset["data"]
        new_data = []
        for ele in data:
            new_ele = {}
            title = ele["title"]
            new_title = title + "_Translated"
            all_paragraphs = ele["paragraphs"]
            new_all_paragraphs = []
            for paragraph in all_paragraphs:
                question_answers = paragraph["qas"]
                answer_idx_list = []
                for qa in question_answers:
                    # question = qa["question"]
                    answers = qa["answers"]
                    answer_start_idx_list = []
                    for answer_ele in answers:
                        answer_start = answer_ele["answer_start"]
                        answer_start_idx_list.append(answer_start)
                    answer_idx_list.append(answer_start_idx_list)
                context = paragraph["context"]
                context_sentences = sent_tokenize(context)
                new_paragraph = {}
                new_context = []
                new_qas = []
                sent_idx = 0
                fwd_ctr = 0
                for sentence in context_sentences:
                    translated_sentence_options = dict.get(sentence.strip())
                    if (translated_sentence_options is None) or fwd_ctr > 0:
                        translated_sentence_options = [sentence]
                        fwd_ctr -= 1
                    translated_sentence = random.choice(translated_sentence_options)
                    new_context.append(translated_sentence.strip())
                    sentence_start_idx = context.index(sentence)
                    sentence_end_idx = sentence_start_idx + len(sentence)
                    answers_idx = 0
                    for answer_start_idxs in answer_idx_list:
                        answer_idx = 0
                        new_question = question_answers[answers_idx]["question"]
                        new_answers = []
                        for answer_start_idx in answer_start_idxs:
                            if answer_start_idx < sentence_end_idx and sentence_start_idx < answer_start_idx:
                                new_answer_ele = {}
                                answers = question_answers[answers_idx]["answers"][answer_idx]
                                answer = answers["text"]
                                intermediate_context = " ".join(new_context).strip()
                                # Answer Index Match
                                intermediate_context_corr_answer = intermediate_context[answer_start_idx: answer_start_idx + len(answer)]
                                if answer in intermediate_context_corr_answer:
                                    new_answer_start_idx, new_answer = answer_start_idx, answer
                                    new_answer_ele["answer_start"] = new_answer_start_idx
                                    new_answer_ele["text"] = new_answer
                                    new_answers.append(new_answer_ele)
                                # Exact Answer Match
                                elif intermediate_context.find(answer) != -1:
                                    new_answer_start_idx, new_answer = intermediate_context.find(answer), answer
                                    new_answer_ele["answer_start"] = new_answer_start_idx
                                    new_answer_ele["text"] = new_answer
                                    new_answers.append(new_answer_ele)
                                # Search for new answer
                                else:
                                    new_answer_start_idx, new_answer = self.find_new_answer(translated_sentence, answer, intermediate_context)
                                    if new_answer_start_idx == -1 and new_answer == -1:
                                        # new_context.remove(translated_sentence)
                                        last_popped_item = new_context.pop()
                                        new_context.append(sentence.strip())
                                        intermediate_context = " ".join(new_context).strip()
                                        new_answer_start_idx = intermediate_context.find(answer)
                                        # Handling for when the sentence tokenizer breaks an answer into 2 sentences
                                        # Such as "Rev. William Corby" is split into "Rev." and "William Corby"
                                        if new_answer_start_idx == -1:
                                            # Try using next sentence to recreate original sentence before tokenizing
                                            ctr = 1
                                            while sent_idx + ctr < len(context_sentences):
                                                intermediate_context += " " + context_sentences[sent_idx+ctr]
                                                new_answer_start_idx = intermediate_context.find(answer)
                                                if new_answer_start_idx == -1:
                                                    # Try adding line without space
                                                    intermediate_context += context_sentences[sent_idx+ctr]
                                                    new_answer_start_idx = intermediate_context.find(answer)
                                                    if new_answer_start_idx == -1:
                                                        # Try adding another line to intermediate context
                                                        ctr += 1
                                                        continue
                                                # Answer has been found
                                                break
                                            fwd_ctr = ctr
                                        new_answer = answer
                                    new_answer_ele["answer_start"] = new_answer_start_idx
                                    new_answer_ele["text"] = new_answer
                                    new_answers.append(new_answer_ele)
                            answer_idx += 1
                        if len(new_answers) > 0:
                            new_qa = {}
                            new_qa["question"] = new_question
                            new_qa["id"] = hash(new_question)
                            new_qa["answers"] = new_answers
                            new_qas.append(new_qa)
                        answers_idx += 1
                    sent_idx += 1
                new_context = " ".join(new_context).strip()
                new_paragraph["context"] = new_context

                # Fix previous answers if required
                for prev_qa in new_qas:
                    for previous_answer in prev_qa["answers"]:
                        prev_start_idx = previous_answer["answer_start"]
                        prev_ans = previous_answer["text"]
                        poss_start_idx = new_context.find(prev_ans)
                        # Check for updated answers from actual answer list if index not found
                        if poss_start_idx == -1:
                            actual_answer = prev_ans
                            for poss_qa in question_answers:
                                if poss_qa["question"] != prev_qa["question"]:
                                    continue
                                actual_answer = poss_qa["answers"][0]["text"]
                                break
                            previous_answer["text"] = actual_answer
                            previous_answer["answer_start"] = new_context.find(actual_answer)
                            if new_context.find(actual_answer) == -1:
                                cleaned_answer = re.sub(r'[^a-zA-Z0-9 ,!?.]', r'', actual_answer)
                                if actual_answer == "\"Boy, Up!\" or \"Boy, Queue!\"":
                                    cleaned_answer = "Boy, Up!\" or \"Boy, Queue!"
                                word_tok = word_tokenize(cleaned_answer)
                                word_ctr = 1
                                while word_ctr < len(word_tok):
                                    search_str = " ".join(word_tok[:-word_ctr])
                                    if new_context.find(search_str) != -1:
                                        previous_answer["answer_start"] = new_context.find(search_str)
                                        previous_answer["text"] = cleaned_answer
                                        break
                                    word_ctr += 1
                        elif poss_start_idx != prev_start_idx:
                            previous_answer["answer_start"] = new_context.index(prev_ans)

                new_paragraph["qas"] = new_qas
                new_all_paragraphs.append(new_paragraph)
            new_ele["paragraphs"] = new_all_paragraphs
            new_ele["title"] = new_title
            new_data.append(new_ele)
        data.extend(new_data)
        new_dataset = {}
        new_dataset["data"] = data
        aug_file = open("train-v1.1-aug.json", "w")
        json.dump(new_dataset, aug_file)
        aug_file.close()


    def find_new_answer(self, translated_sentence, orig_answer, intermediate_context):
        ngram_comparator = NGram(N=2)
        translated_tokens = word_tokenize(translated_sentence)
        answer_tokens = word_tokenize(orig_answer)

        new_answer = []

        for answer_token in answer_tokens:
            best_ngram_score = 0.0
            best_translated_answer_token = answer_token
            for translated_token in translated_tokens:
                curr_ngram_score = ngram_comparator.compare(answer_token, translated_token)
                if curr_ngram_score > best_ngram_score:
                    best_ngram_score = curr_ngram_score
                    best_translated_answer_token = translated_token
            new_answer.append(best_translated_answer_token)

        new_answer = " ".join(new_answer).strip()
        new_answer_idx = intermediate_context.find(new_answer)
        if new_answer_idx != -1:
            return new_answer_idx, new_answer
        else:
            return -1, -1


if __name__ == '__main__':
    DataAugmentor().create_augmented_data()