from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

ps = ["Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"]

# Completion API
stream = False
completion = client.completions.create(
    model=model,
    prompt=ps,
    echo=False,
    temperature=0,
    stream=stream)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
for i, c in enumerate(completion.choices):
    print(f"Prompt: {ps[i]!r}, Generated text: {c.text!r}")
# print(completion.tokens)