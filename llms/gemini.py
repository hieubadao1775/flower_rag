class Gemini:
    def __init__(self,
                 model_name="gemini-2.5-flash",
                 temperature=0.7,
                 top_p=0.9):
        from google import genai

        self.client = genai.Client()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def answer(self, history_chat: list) -> str:
        from google.genai import types

        system_prompt = history_chat[0]["content"]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            top_p=self.top_p
        )

        gemini_history = []
        for message in history_chat[1:]:
            if message["role"] == "user":
                gemini_history.append({"role": "user", "parts": [{"text": message["content"]}]})
            elif message["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [{"text": message["content"]}]})

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_history,
            config=config,
        )

        return response.text


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm = Gemini()
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

