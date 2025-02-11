import openai

# openai.api_base = "https://api.openai.com/v1" # 换成代理，一定要加v1
openai.api_base = "https://openkey.cloud/v1" # 换成代理，一定要加v1
# openai.api_key = "API_KEY"
openai.api_key = "sk-J6LW3tnRjWJhZtQiB69bE9C3E7554594A6F45f21884a0a7a"

#这里写的是裁判的prompt
PROMPT_TEMPLATE='''
Sentence:
The white mouse is on the left of the keyboard.

Entities:
mouse.keyboard

Questions:
What color is the mouse?&mouse
Is the white mouse on the left of the keyboard?&mouse.keyboard

Sentence:
{sent}

Entities:
{entity}

Questions:'''


sent = "yes, the blue umbrella is under the black umbrella in the image."
entity = "umbrella"


content = PROMPT_TEMPLATE.format(sent=sent, entity=entity)

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{
        'role': 'system',
        'content': 'You are a language assistant that helps to ask questions about a sentence.'
    }, {
        'role': 'user',
        'content': content,
    }],
    temperature=0.2,
    max_tokens=1024,
)
res = response['choices'][0]['message']['content'].splitlines()
print(res)