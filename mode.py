import pickle
from classification.bert import predict_genre
from transformers import BertTokenizerFast, BertForSequenceClassification


with open('classification/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('gyubinc/bert-book32-gyubin', num_labels=32)


if __name__ == "__main__":
    # predict_genre()
    text = """
    This night, the fireflies of the day are yours
    I'll send it near the window
    I mean I love you
    I remember our first kiss
    So anytime close your eyes
    go to the furthest place
    I was a wave
    Like letters written in the sand
    I think you will disappear far away
    I always miss you, miss you
    all the words here in my heart
    I can't take it all out
    I mean I love you
    how to me
    Have you been lucky?
    if we are together now
    oh how nice
    I was a wave
    Like letters written in the sand
    I think you will disappear far away
    I miss you again, I miss you more
    all the words in my diary
    I can't take it all out
    saying I love you
    """

    # 텍스트의 장르 예측
    music_genre = predict_genre(text, model, tokenizer, le, k=1)  # 'le'를 사용하셔야 합니다.
    print(music_genre)