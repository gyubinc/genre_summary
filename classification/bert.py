from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd


label_dict = {
    'Arts & Photography': ['Dramedy', 'Ethereal', 'Soft Rock'],
    'Biographies & Memoirs': ['Jazz', 'Lively', 'Swing'],
    'Business & Money': ['Ambient', 'Minimalism', 'Techno'],
    'Calendars': ['Acoustic', 'Lullaby', 'Piano'],
    "Children's Books": ['Children', 'Lullaby', 'Piano'],
    'Comics & Graphic Novels': ['Industrial', 'Punk', 'Rocktronica'],
    'Computers & Technology': ['EDM', 'Glitch House', 'Synth Pop'],
    'Cookbooks, Food & Wine': ['Jazz', 'Lounge', 'Smooth Jazz'],
    'Crafts, Hobbies & Home': ['Folk', 'Indie Folk', 'Lively'],
    'Christian Books & Bibles': ['Gospel', 'Hymns', 'Inspirational'],
    'Engineering & Transportation': ['Orchestral', 'Racing', 'Suspense'],
    'Health, Fitness & Dieting': ['Electronic', 'Motivational', 'Progressive House'],
    'History': ['Classical', 'Orchestral', 'Piano'],
    'Humor & Entertainment': ['Jazz', 'Lounge', 'Quirky'],
    'Law': ['Blues', 'Reggae', 'Suspense'],
    'Literature & Fiction': ['Cinematic', 'Fantasy', 'Orchestral'],
    'Medical Books': ['Ambient', 'Electronic', 'Synth Pop'],
    'Mystery, Thriller & Suspense': ['Electronic', 'Suspense', 'Synth Pop'],
    'Parenting & Relationships': ['Ballad', 'Lullaby', 'Piano'],
    'Politics & Social Sciences': ['Blues', 'Orchestral', 'Suspense'],
    'Reference': ['Jazz', 'Lounge', 'World Elements'],
    'Religion & Spirituality': ['Gospel', 'Hymns', 'Inspirational'],
    'Romance': ['Ballad', 'Electronic', 'Hawaiian'],
    'Science & Math': ['Electronic', 'Experimental', 'Synth Pop'],
    'Science Fiction & Fantasy': ['Cinematic', 'Fantasy', 'Orchestral'],
    'Self-Help': ['Ambient', 'Meditation', 'New Age'],
    'Sports & Outdoors': ['Electronic', 'Rock', 'Sports'],
    'Teen & Young Adult': ['Indie Pop', 'Lively', 'Pop'],
    'Test Preparation': ['Action', 'Epic', 'Trailer'],
    'Travel': ['Folk', 'Lively', 'World'],
    'Gay & Lesbian': ['Dance', 'Electronic', 'Pop'],
    'Education & Teaching': ['Children', 'Lively', 'Piano']
}



def predict_genre(text, model, tokenizer, label_encoder, k=3):
    # 디바이스 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 모델을 올바른 디바이스로 이동
    model.to(device)

    # 텍스트를 토큰화합니다
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device)

    # 예측을 수행합니다
    outputs = model(**inputs)

    # 가장 높은 확률을 가진 클래스를 찾습니다
    topk_values, topk_indices = torch.topk(outputs.logits, k, dim=-1)

    # 클래스를 원래의 텍스트 라벨로 변환합니다
    predicted_labels = label_encoder.inverse_transform(topk_indices.cpu().numpy()[0])
    
    if k == 1:
        music_genre = label_dict[predicted_labels[0]]
        rand = np.random.choice(music_genre, size=2, replace=False)
        print(f'예상 book genre = {predicted_labels[0]}')
        print(f'해당 book genre에 대한 음악 genre = {music_genre}')
        print(f'random한 2개 요소 = {rand}')
        return rand
    
    return predicted_labels  # 결과가 리스트인데 첫 번째 요소만 반환합니다.