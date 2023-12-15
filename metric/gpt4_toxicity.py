import openai
openai.api_key = "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"

# import pdb;pdb.set_trace()
client = openai.OpenAI(api_key=openai.api_key)
response = client.moderations.create(input="Sample text goes here.")

output = response.results[0]
