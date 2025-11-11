from httpx import Response
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback
load_dotenv()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)


def get_current_weather(location: str, unit: str):
    print(f"[debug] getting weather for {location} in {unit}")
    return f"The weather in {location} is 72 degrees {unit} and sunny."


# Define tools
tools = [
    {
        "type": "code_interpreter",
        "container": {
            "type": "auto"
        }
    },
    {
        "type": "web_search"
    },
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        }
    },
]

# Create a running input list we will add to over time
input_list = [
    {"role": "system", "content": "You are a helpful assistant that can use the browser and python tools to answer questions."},
    {"role": "user", "content": "What is the weather in san francisco and will Russia capture Lyman by December 31? Make sure to use the browser tool to research the second question."}
]

# 1. First API call with the user's question
print("=" * 70)
print("Making first API call...")
try:
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=input_list,
        reasoning={
            "effort": "medium",
            "summary": "detailed"
        },
        tools=tools,
        extra_body={"enable_response_messages": True}
    )
except Exception as e:
    print(traceback.format_exc())
    exit(1)

print("First response output:")
for item in response.output:
    print(f"  - {item.type}: {item}")

# Save the first response to JSON for inspection
with open("response.json", "w") as f:
    json.dump(response.model_dump(mode="json"), f, indent=4)

# 2. Save function call outputs and execute them
input_list += response.output

available_tools = {"get_current_weather": get_current_weather}

for item in response.output:
    if item.type == "function_call":
        print(f"\n[Executing function call: {item.name}]")

        # Execute the function
        tool_to_call = available_tools[item.name]
        args = json.loads(item.arguments)
        result = tool_to_call(**args)

        print(f"Function result: {result}")

        # 3. Provide function call results back to the model
        input_list.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result
        })
        
        
if hasattr(response, 'output_text'):
    print("Output text:")
    print(response.output_text)

# 4. Make second API call with function results
print("\n" + "=" * 70)
print("Making second API call with function results...")
response_2 = client.responses.create(
    model="openai/gpt-oss-20b",
    input=input_list,
    reasoning={
        "effort": "low",
        "summary": "detailed"
    },
    tools=tools,
    extra_body={"enable_response_messages": True}
)

print("=" * 70)
print("\nFinal response:")
print(response_2.model_dump_json(indent=2))
print("\n" + "=" * 70)
if hasattr(response_2, 'output_text'):
    print("Output text:")
    print(response_2.output_text)
else:
    print("Output:")
    for item in response_2.output:
        if hasattr(item, 'text'):
            print(item.text)