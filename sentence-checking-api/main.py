from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import nltk
from nltk.corpus import words as nltk_words
from urllib.parse import unquote_plus

# Download required nltk corpus only once
nltk.download('words', quiet=True)
word_list = set(nltk_words.words())

# Load models once
try:
    sentiment_model = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )

    ner_model = pipeline(
        "ner",
        model="dslim/bert-base-NER",  # Lighter model than bert-large
        aggregation_strategy="simple",
        device=-1  # CPU
    )
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# FastAPI app
app = FastAPI()

# Request model
class SentenceRequest(BaseModel):
    sentence: str

# Main sentence analysis logic
def analyze_sentence(sentence: str):
    try:
        sentence = unquote_plus(sentence).strip()
        if not sentence:
            raise ValueError("Empty sentence")

        # Sentiment
        sentiment_result = sentiment_model(sentence)[0]
        sentiment_score = round(float(sentiment_result['score']), 4)

        # NER
        entities_raw = ner_model(sentence)
        entities = [
            {
                "entity": ent['entity_group'],
                "word": ent['word'],
                "score": round(float(ent['score']), 4)
            }
            for ent in entities_raw
        ]

        # Word-based processing
        words_in_sentence = [word.strip('.,!?') for word in sentence.split()]
        keywords = [word for word in words_in_sentence if len(word) > 4]
        valid_words = [w for w in words_in_sentence if w.lower() in word_list]
        is_meaningful = len(valid_words) >= 2

        return {
            "sentence": sentence,
            "is_meaningful": is_meaningful,
            "sentiment": {
                "label": sentiment_result['label'],
                "confidence": sentiment_score
            },
            "entities": entities,
            "keywords": keywords
        }

    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error processing sentence: {str(ex)}")

# POST method
@app.post("/check-sentence/")
async def check_sentence_post(request: SentenceRequest):
    return analyze_sentence(request.sentence)

# GET method
@app.get("/check-sentence/{sentence}")
async def check_sentence_get(sentence: str):
    return analyze_sentence(sentence)
