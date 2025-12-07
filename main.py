from openai import OpenAI

conv = []

def hello_world():
    print("hello nigga")
    return "called and printed to the user"


def handle_tools():
    tools = conv[-1].tool_calls
    if tools:
        for tool in tools:
            match tool.function.name:
                case "hello_world":
                    conv.append({"role": "tool", "tool_call_id": tool.id, "content": hello_world()})

    

def main():

    client = OpenAI(
        api_key="<api-key>",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "hello_world",
                "description": "Call this function when the user asks to say hello world.",
                "parameters": {
                    "type": "object",
                    "properties": {},  # No parameters needed for this simple tool
                    "required": [],
                },
            },
        }
    ]

    while True:

        conv.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": input("Enter prompt: ")
                }
            ])

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages = conv,
            tools=tools,
            tool_choice="auto"
        )

        conv.append(response.choices[0].message)
        handle_tools()

        content = response.choices[0].message.content
        if(content):
            print(f"Model response: {content}")


if __name__ == "__main__":
    main()