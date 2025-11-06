from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="http://localhost:8000/v1"
    )

response = client.responses.create(
    model="openai/gpt-oss-20b",
    input="Search for trellis in san francisco.",
    reasoning={
        "effort": "low",
        "summary": "detailed"
    },
    tools=[
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto"
            }
        },
        {
            "type": "web_search_preview"
        }
    ],
    extra_body={"enable_response_messages": True}
)

import json
with open("response.json", "w") as f:
    json.dump(response.model_dump(mode="json"), f, indent=4)