from collections import defaultdict
import os
import json
import asyncio
import aiohttp
import pandas as pd

import openai
#openai.api_key = 'YOUR_KEY_HERE'


from data import *



async def generate_explanation_chatgpt(original, edit):
    prompt = """
        The following news headlines have been edited to be more humorous.
        The format of the headline is "text text [[ original word => edited word ]] text text".
        Explain what kind of humorous response the edit wanted to elicit, and wether it suceeeded or fell flat.
        You are not to be too easily offended. Answer as concisely as possible. When explaining something refer to the exact part in the headline.
        Do not use more than 3 sentences. Only output the explanation, nothing else.

        Headline:
        REPLACE_WITH_HEADLINE
    """

    combined = original.replace("<", "[ ").replace("/>", f" => {edit} ]")
    this_prompt = prompt.replace("REPLACE_WITH_HEADLINE", combined)
    
    while True:
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    { "role": "user", "content": this_prompt }
                ],
                temperature=0.7,
                max_tokens=250
            )
            return original, edit, [choice["message"]["content"] for choice in response['choices']]
        except Exception as e:
            print(e)
            print("retrying...")
            await asyncio.sleep(5)

async def generate_explanations_chatgpt(df, filename, starting_idx=0):
    completions = defaultdict(list)
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        openai.aiosession.set(session)
        
        i = starting_idx  
        while i < len(df.index):
            initial_idx = i
            tasks = []
            while i < min(initial_idx + 150, len(df.index)):
                headline = df.iloc[i]
                original = headline["original"]
                edit = headline["edit"]
                print("generating", i, "/", len(df.index))

                task = generate_explanation_chatgpt(session, i, original, edit)
                tasks.append(task)
                i += 1

            responses = await asyncio.gather(*tasks)

            for original, edit, choices in responses:
                for choice in choices:
                    completions[headline_key(original, edit)].append(choice)

            with open(filename + f"checkpoint{initial_idx}.json", "w") as f:
                json.dump(completions, f, indent=4)

            await asyncio.sleep(60)
        
        with open(filename + f"checkpoint{initial_idx}.json", "w") as f:
            json.dump(completions, f, indent=4)

    return completions

async def generate_explanation_vicuna(session, original, edit):
    prompt = """
Explain this humorous/satirical news headline edit to a foreginer who doesn't know the context well and whose first language isn't English. After your explanation they should be able to understand the meaning of the joke - what makes it funny and clever, but also which parts fall flat to native speakers. Be succint and direct, make only specific statements.

        Headline:
        REPLACE_WITH_HEADLINE
    """

    combined = original.replace("<", "[ ").replace("/>", f" => {edit} ]")
    this_prompt = prompt.replace("REPLACE_WITH_HEADLINE", combined)
    
    request = {
        'user_input': this_prompt,
        'history': {'internal': [], 'visible': []},
        'mode': 'instruct',
        'character': 'Example',
        'instruction_template': 'Vicuna-v1.1',
        'regenerate': False,
        '_continue': False,
        'stop_at_newline': True,
        'chat_prompt_size': 2048,
        'chat_generation_attempts': 1,
        'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "".\n\n',
        'max_new_tokens': 250,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': True,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': ['\n###']
    }
    
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://127.0.0.1:5000/api/v1/chat", json=request) as response:
                    if response.status == 200:
                        result = await response.json()
                        return original, edit, [result['results'][0]['history']['visible'][-1][1]]
        except Exception as e:
            print(e)
            print("retrying...")
            await asyncio.sleep(5)

async def generate_explanations_vicuna(df, filename, starting_idx=0):
    completions = defaultdict(list)
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        openai.aiosession.set(session)
        
        i = starting_idx  
        while i < len(df.index):
            initial_idx = i
            tasks = []
            while i < min(initial_idx + 1, len(df.index)):
                headline = df.iloc[i]
                original = headline["original"]
                edit = headline["edit"]
                print("generating", i, "/", len(df.index))

                task = generate_explanation_vicuna(session, i, original, edit)
                tasks.append(task)
                i += 1

            responses = await asyncio.gather(*tasks)

            for original, edit, choices in responses:
                for choice in choices:
                    completions[headline_key(original, edit)].append(choice)

            if i % 100 == 0:
                with open(filename + f"checkpoint{initial_idx}.json", "w") as f:
                    json.dump(completions, f, indent=4)
        
        with open(filename + f"checkpoint{initial_idx}.json", "w") as f:
            json.dump(completions, f, indent=4)

    return completions


def merge_explanations(dir, prefix, new_file_path):
    explanations = {}
    for f in os.listdir(dir):
        if not f.startswith(prefix):
            continue
        with open(os.path.join(dir, f), "r") as f:
            explanations.update(json.load(f))
    
    with open(new_file_path, "w") as f:
        json.dump(explanations, f, indent=4)

    print("merged explanations:", len(explanations))



if False:    
    train_df, valid_df, test_df = make_base_dataset()

    os.makedirs("explanations_vicuna", exist_ok=True)
    asyncio.run(generate_explanations_vicuna(train_df, "explanations_vicuna_checkpoints/train_",))
    asyncio.run(generate_explanations_vicuna(valid_df, "explanations_vicuna_checkpoints/valid_",))
    asyncio.run(generate_explanations_vicuna(test_df,  "explanations_vicuna_checkpoints/test_"))

    os.makedirs("explanations", exist_ok=True)
    merge_explanations("explanations_vicuna_checkpoints", "train_", "explanations/vicuna_train.json")
    merge_explanations("explanations_vicuna_checkpoints", "valid_", "explanations/vicuna_valid.json")
    merge_explanations("explanations_vicuna_checkpoints", "test_", "explanations/vicuna_test.json")

if False:    
    train_df, valid_df, test_df = make_base_dataset()

    os.makedirs("explanations_chatgpt", exist_ok=True)
    asyncio.run(generate_explanations_chatgpt(train_df, "explanations_chatgpt_checkpoints/train_"))
    asyncio.run(generate_explanations_chatgpt(valid_df, "explanations_chatgpt_checkpoints/valid_"))
    asyncio.run(generate_explanations_chatgpt(test_df,  "explanations_chatgpt_checkpoints/test_"))

    os.makedirs("explanations", exist_ok=True)
    merge_explanations("explanations_chatgpt_checkpoints", "train_", "explanations/chatgpt_train.json")
    merge_explanations("explanations_chatgpt_checkpoints", "valid_", "explanations/chatgpt_valid.json")
    merge_explanations("explanations_chatgpt_checkpoints", "test_", "explanations/chatgpt_test.json")