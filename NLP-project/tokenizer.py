from transformers import AutoTokenizer
from utils import process_train
def find_token_position(texts, queries, answers, model_pretrained = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_pretrained)
    encodings = tokenizer(texts, queries, max_length = 512,truncation=True, padding='max_length')

    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
      
        # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            # if end position is still None the answer passage has been truncated
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

    # Update the data in dictionary
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    return encodings

if __name__ == '__main__':
    train_texts, train_queries, train_answers = process_train()

    train_texts = train_texts[:10] 
    train_queries = train_queries[:10]
    train_answers = train_answers[:10]

    train_encodings = find_token_position(train_texts, train_queries, train_answers)

    print(train_encodings['start_positions'])
    print(train_encodings.keys())



    
