from ai_engines.openai_engine import get_openai_response
from ai_engines.claude_engine import get_claude_response
import json

def parse_ai_response(text):
    try:
        response = text.split("Response:")[1].split("Tag:")[0].strip()
        tag = text.split("Tag:")[1].strip()
        return {"response": response, "tag": tag}
    except:
        return {"response": "Sorry, couldn't parse the AI reply.", "tag": "unknown"}

def generate_best_response(user_input):
    res_openai = get_openai_response(user_input)
    res_claude = get_claude_response(user_input)

    ai_responses = []
    if res_openai:
        parsed = parse_ai_response(res_openai)
        parsed['source'] = 'openai'
        ai_responses.append(parsed)

    if res_claude:
        parsed = parse_ai_response(res_claude)
        parsed['source'] = 'claude'
        ai_responses.append(parsed)

    if not ai_responses:
        return None

    best = max(ai_responses, key=lambda x: len(x["response"]))
    print(f"[AI Source Used: {best['source']}]")
    return {"tag": best["tag"], "response": best["response"]}
