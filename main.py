from openai import OpenAI


def main():

    client = OpenAI(
        api_key="<api-key>",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": input("Enter prompt: ")
            }
        ]
    )

    print(response.choices[0])

if __name__ == "__main__":
    main()