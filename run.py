from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoTokenizer
from summary.summary import summary
from classification.classification import classification
import joblib
import pandas as pd

def main(passage):
    MODEL_NAME = "google/pegasus-xsum"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    result_sum = summary(tokenizer, model, passage)

    data = pd.read_csv('classification/vector_data.csv',encoding = "ISO-8859-1")
    clf = joblib.load('classification/best.pkl')
    result_clf = classification(data, clf, passage)
    
    print(result_sum)
    print(result_clf)
    return result_sum, result_clf
    


if __name__ == "__main__":
    passage = """\
    As a Queen-Ka, I not only showcased my external beauty but also my inner allure.    
    I possessed an unwavering confidence that effortlessly inspired those around me.
    Through my presence, I encouraged others to embrace their own individuality and find pride in their uniqueness.
    """
    
    main(passage)
    
    