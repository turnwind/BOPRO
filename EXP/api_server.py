import openai
import time

def chat(says,model="gpt-4o-2024-05-13"):
    client = openai.OpenAI(
        base_url="https://api.chatanywhere.tech/v1"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": says,
                }
            ],
            model=model,
        )
        print("bots: "+chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    except openai.Timeout as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(says,model=model)

    except openai.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(says,model=model)

    except openai.APIError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(says,model=model)

    except openai.APIConnectionError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"API connection error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(says,model=model)
    except openai.ServiceUnavailableError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Service unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return chat(says,model=model)

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(
            f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
        )
        time.sleep(retry_time)
        return chat(says,model=model)
