class OllamaHost:
    def __init__(self,
                 model_name="qwen3-claude:latest",
                 temperature=0.7,
                 top_p=0.9):
        from ollama import Client

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.client = Client()

    def answer(self, history_chat: list) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=history_chat,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        )

        return response.message.content


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = OllamaHost(model_name="qwen3-claude")
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
