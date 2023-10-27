from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)


model_path = "C:/Users/mathe/OneDrive/Desktop/projects/NM/backend/fitnessbot/fine_tuned_model.pkl"
with open(model_path, "rb") as model_file:
    model = torch.load(model_file)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@app.route("/generate_response", methods=["POST"])
def generate_response():
    user_input = request.json.get("user_input")

    if not user_input:
        return jsonify({"error": "Please provide a user_input parameter."})


    input_ids = tokenizer.encode("You: " + user_input, return_tensors="pt")
    response = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    bot_response = tokenizer.decode(response[0], skip_special_tokens=True)

    return jsonify({"bot_response": bot_response})

@app.route("/", methods=["GET"])
def health_check():
    return "Health check passed."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
