from flask import Flask, request, jsonify
from analysis.fft_metrics import analyze_images

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json

        original_path = data.get("original")
        transformed_path = data.get("transformed")

        if not original_path or not transformed_path:
            return jsonify({
                "error": "Both 'original' and 'transformed' paths are required."
            }), 400

        results = analyze_images(original_path, transformed_path)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)