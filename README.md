
# Implementation-of-Machine-Translation
## NAME: SANJUSHRI A
## REGISTER NUMBER: 212223040187

## AIM:
To create a simple language translation application that translates English sentences into French using a pre-trained Transformer model (Helsinki-NLP/opus-mt-en-fr).

## PROCEDURE:


1. **Objective**: Translate English to French using a pre-trained Transformer model.
2. **Install Libraries**: Use `pip install transformers torch`.
3. **Import Modules**: Import `AutoTokenizer`, `AutoModelForSeq2SeqLM`, and `torch`.
4. **Load Model**: Load `Helsinki-NLP/opus-mt-en-fr` model and tokenizer.
5. **Define Function**: Create `translate_text()` to handle translation.
6. **Input Sentences**: Write sample English sentences in a list.
7. **Translate**: Loop through each sentence and translate it.
8. **Print Output**: Display original and translated sentences.
9. **Capture Output**: Take screenshots of translated results.
10. **Result**: Confirm that translation is accurate and effective.



## PROGRAM:
```
!pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer for English-to-French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
import torch

def translate_text(text: str, max_length: int = 40) -> str:
    # Tokenize the input text and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate translation using the model
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated IDs back to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
# Sample sentences in English
english_sentences = [
    "Hello, how are you?",
    "This is an experiment in machine translation.",
    "Transformers are powerful models for natural language processing tasks.",
    "Can you help me with my homework?",
    "I love learning new languages."
]

# Translate each sentence
for sentence in english_sentences:
    translation = translate_text(sentence)
    print(f"Original: {sentence}")
    print(f"Translated: {translation}\n")
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/0f1946e6-8c76-41d6-9a86-6d680a432587)

![image](https://github.com/user-attachments/assets/9c5d06c8-d52c-4098-a78c-ee0ce1a10fb3)

![image](https://github.com/user-attachments/assets/35b91135-ef98-466c-82a5-fc92cde61d36)

## RESULT:
The model successfully translated English sentences into French using a pre-trained Transformer model (Helsinki-NLP/opus-mt-en-fr). The output is accurate and demonstrates the power of transfer learning in NLP.


