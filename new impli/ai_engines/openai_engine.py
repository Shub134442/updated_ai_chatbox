import openai
openai.api_key = "YOUR_OPENAI_API_KEY"

def get_openai_response(user_input):
    prompt = f'''
User said: "{user_input}"

1. Suggest a chatbot response.
2. Suggest an intent tag.

Format:
Response: <response>
Tag: <intent_tag>
'''
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message["content"].strip()
    except Exception as e:
        print("[OpenAI] Error:", e)
        return None