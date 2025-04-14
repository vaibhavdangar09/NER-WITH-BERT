from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

app = Flask(__name__)

# Load POS model
pos_tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")
pos_model = AutoModelForTokenClassification.from_pretrained("vblagoje/bert-english-uncased-finetuned-pos")
pos_pipeline = pipeline("token-classification", model=pos_model, tokenizer=pos_tokenizer)

# Load NER model (raw, for better control)
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline_raw = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, grouped_entities=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]

    # Tokenize input manually
    encoding = ner_tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0]

    # Get token-level words
    tokens = ner_tokenizer.convert_ids_to_tokens(input_ids)

    # Run raw NER and POS predictions
    ner_outputs = ner_pipeline_raw(text)
    pos_outputs = pos_pipeline(text)

    # NER: Reconstruct merged tokens (handle subwords)
    words = []
    ner_tags = []
    pos_tags = []

    current_word = ""
    current_ner = "O"
    ner_index = 0
    pos_index = 0

    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                # Commit previous word
                words.append(current_word)
                ner_tags.append(current_ner)
                if pos_index < len(pos_outputs):
                    pos_tags.append(pos_outputs[pos_index]["entity"])
                else:
                    pos_tags.append("X")
                pos_index += 1
            current_word = token
            current_ner = "O"

            # Try to align NER
            for j in range(ner_index, len(ner_outputs)):
                ner_token = ner_outputs[j]
                if ner_token["word"].replace("##", "") in token.replace("##", ""):
                    current_ner = ner_token["entity_group"] if "entity_group" in ner_token else ner_token["entity"]
                    ner_index = j + 1
                    break

    # Commit final word
    if current_word:
        words.append(current_word)
        ner_tags.append(current_ner)
        if pos_index < len(pos_outputs):
            pos_tags.append(pos_outputs[pos_index]["entity"])
        else:
            pos_tags.append("X")

    return jsonify({
        "tokens": words,
        "ner": ner_tags,
        "pos": pos_tags
    })


if __name__ == "__main__":
    app.run(debug=True)
