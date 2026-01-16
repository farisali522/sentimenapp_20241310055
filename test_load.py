import pickle
try:
    with open('model_uas_20241310055.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer_uas_20241310055.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("SUCCESS: Model and Vectorizer loaded correctly.")
    print("Model classes:", model.classes_)
except Exception as e:
    print(f"FAILURE: {e}")
