from src.predict_sentiment import predict_sentiment

if __name__ == "__main__":
    print("=== Opinion Mining & Sentiment Analysis ===")
    while True:
        user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")
