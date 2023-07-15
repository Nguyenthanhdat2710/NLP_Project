import json
from pathlib import Path
def process_data(data):
    texts = []
    queries = []
    answers = []
      # Search for each passage, its question and its answer
    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    # Store every passage, query and its answer to the lists
                    texts.append(context)
                    queries.append(question)
                    answers.append(answer)
    return texts, queries, answers


def end_position_character(texts, answers):
    for answer, text in zip(answers, texts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        # Get the real end index
        end_idx = start_idx + len(real_answer)

        # Deal with the problem of 1 or 2 more characters 
        if text[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        # When the real answer is more by one character
        elif text[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  
        # When the real answer is more by two characters  
        elif text[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2
    return  texts, answers

def process_train(path = Path('squad/train-v2.0.json')):
    with open(path, 'rb') as f:
        squad_dict_train = json.load(f)
    train_texts, train_queries, train_answers = process_data(squad_dict_train)
    train_texts, train_answers = end_position_character(train_texts, train_answers)
    return train_texts, train_queries, train_answers 
def process_val(path = Path('squad/dev-v2.0.json')):
    with open(path, 'rb') as f:
        squad_dict_val = json.load(f)
    val_texts, val_queries, val_answers = process_data(squad_dict_val)
    val_texts, val_answers = end_position_character(val_texts, val_answers)
    return val_texts, val_queries, val_answers
if __name__ == '__main__':
    train_texts, train_queries, train_answers = process_train()
    val_texts, val_queries, val_answers = process_val()
    print(train_texts[0])
    print(train_queries[0])
    print(train_answers[0])
    print(val_answers[0])