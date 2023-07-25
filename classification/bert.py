from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd


label_dict = {
    'Arts & Photography': ['Smooth Rock', 'Ethereal', 'Minimalism', 'Soft Rock', 'Ambient'],
    'Biographies & Memoirs': ['Lively', 'Swing', 'Jazz', 'Ballad', 'Piano'],
    'Business & Money': ['Glitch House', 'Techno', 'Ambient', 'EDM', 'Minimalism'],
    'Calendars': ['Acoustic', 'Lullaby', 'Piano', 'New Age', 'Meditation'],
    "Children's Books": ['Children', 'Lullaby', 'Piano', 'Folk', 'World Music'],
    'Comics & Graphic Novels': ['Rocktronica', 'Industrial', 'Punk', 'Heavy Metal', 'Blues'],
    'Computers & Technology': ['Synth Pop', 'EDM', 'Glitch House', 'Electronic', 'Techno'],
    'Cookbooks, Food & Wine': ['Smooth Jazz', 'Lounge', 'Jazz', 'Blues', 'Soul'],
    'Crafts, Hobbies & Home': ['Folk', 'Indie Folk', 'Lively', 'Acoustic', 'Soft Rock'],
    'Christian Books & Bibles': ['Gospel', 'Hymns', 'Inspirational', 'Christian Music', 'Piano'],
    'Engineering & Transportation': ['Suspense', 'Orchestral', 'Racing', 'Electronic', 'Cinematic'],
    'Health, Fitness & Dieting': ['Progressive House', 'Electronic', 'Motivational', 'Ambient', 'New Age'],
    'History': ['Classical', 'Orchestral', 'Piano', 'Cinematic', 'Dramedy'],
    'Humor & Entertainment': ['Jazz', 'Lounge', 'Quirky', 'Comedy', 'Funk'],
    'Law': ['Blues', 'Suspense', 'Reggae', 'Electronic', 'Cinematic'],
    'Literature & Fiction': ['Cinematic', 'Orchestral', 'Fantasy', 'Piano', 'Dramedy'],
    'Medical Books': ['Ambient', 'Synth Pop', 'Electronic', 'New Age', 'Meditation'],
    'Mystery, Thriller & Suspense': ['Suspense', 'Electronic', 'Synth Pop', 'Blues', 'Cinematic'],
    'Parenting & Relationships': ['Ballad', 'Lullaby', 'Piano', 'Acoustic', 'Folk'],
    'Politics & Social Sciences': ['Orchestral', 'Blues', 'Suspense', 'Cinematic', 'Jazz'],
    'Reference': ['World Elements', 'Jazz', 'Lounge', 'Ethnic', 'Acoustic'],
    'Religion & Spirituality': ['Gospel', 'Hymns', 'Inspirational', 'Christian Music', 'Piano'],
    'Romance': ['Hawaiian', 'Ballad', 'Electronic', 'Pop', 'Acoustic'],
    'Science & Math': ['Electronic', 'Experimental', 'Synth Pop', 'Ambient', 'New Age'],
    'Science Fiction & Fantasy': ['Cinematic', 'Orchestral', 'Fantasy', 'Piano', 'Dramedy'],
    'Self-Help': ['Ambient', 'Meditation', 'New Age', 'Soft Rock', 'Piano'],
    'Sports & Outdoors': ['Sports', 'Electronic', 'Heavy Metal', 'Rock', 'Punk'],
    'Teen & Young Adult': ['Pop', 'Indie Pop', 'Lively', 'Acoustic', 'Soft Rock'],
    'Test Preparation': ['Trailer', 'Epic', 'Cinematic', 'Action', 'Dramatic'],
    'Travel': ['World', 'Lively', 'Folk', 'Ethnic', 'Acoustic'],
    'Gay & Lesbian': ['Pop', 'Dance', 'Electronic', 'Soul', 'R&B'],
    'Education & Teaching': ['Lively', 'Children', 'Piano', 'Acoustic', 'Soft Rock']
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