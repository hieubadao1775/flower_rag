class DeepSeek:
    def __init__(self,
                 model_name="deepseek-chat",
                 temperature=0.7,
                 top_p=0.9):
        from openai import OpenAI
        import os

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    def answer(self, chat_history: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p

        ).choices[0].message.content
        return response


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = DeepSeek()
    history_chat = [
        {"role": "system", "content": "You are helpful assistant."}
    ]

    while True:
        query = input("User:\n")
        if query == "quit":
            break
        history_chat.append({"role": "user", "content": query})
        response = llm.answer(history_chat)
        history_chat.append({"role": "assistant", "content": response})
        print("Assistant:\n", response)
