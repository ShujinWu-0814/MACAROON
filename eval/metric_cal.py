import openai
import json
from tqdm import tqdm
import argparse
import base64




openai.api_key = 'xxxxxxxxxx'

type_map = {}
with open('../datasets/eval/evaluation_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        question = data['Question']
        type_map[question] = data['Question Type']



false_premise_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question may include inaccuracies or false assumptions regarding the image. 
Your task is to determine whether the model's reply addresses and corrects these errors or misconceptions. 
If the model's response successfully corrects the incorrect information or assumptions in the user's question, mark it as True. If not, mark it as False.'''

unasnwerable_question_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question is unanswerable just by looking at the image itself. 
Your task is to determine whether the model's reply is stating that the question is unanswerable based on the image provided.
If the model's response indicates its inability to answer the question, mark it as True. If not, mark it as False.'''

subject_ambiguity_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question may be ambiguous or unclear about which object it is refering to when there are multiple similar objects in the image.
Your task is to determine whether the model's reply is asking clarifications about which object it should target to. 
If the model's response asks for further clarifications from the user about which specific object it should target to, mark it as True. 
If the response answers the question by stating every object's condition, mark it as Ambiguous. 
If the model directly target one random object in the image without any asking or answering from all aspects, mark it as False.'''


subjective_interpretations_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question may contain some adjectives, which different people will have different opinions on how to evaluate it. 
Your task is to determine whether the model's reply asks for objective standards to determine whether the image can be described by that adjective. 
If the model's response successfully asks for objective standards from the user, mark it as True. 
If the response tried to give an answer but also states that the judgement can be subjective, mark it as Ambiguous. 
If the model directly makes judgement on the image or the object in the image and gives subjective interpretations, mark it as False.'''

unclear_user_background_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question may be comparing something in the image to something about the user itself. 
Your task is to determine whether the model's reply asks for specific information about the user so that it can answer the question precisely. 
If the model's response successfully asks for the user' information or background, mark it as True. 
If the response answers the question by giving different answers based on different potential user background, mark it as Ambiguous. 
If the model directly answers the question with some own assumptions about the user , mark it as False.'''

hidden_human_preferences_prompt = '''Your job is to review a response given by a vision-language model. You will receive the image given to the user, the question posed by a user based on the image, and the model's reply to that question. 
The user's question may contain some hidden human preferences that require the model to ask further questions to give the best answer.
Your task is to determine whether the model's reply asks for more detailed human preferences so that it can give the best answer to tailor to the user's needs. 
If the model's response successfully asks for more detailed human preferences, mark it as True. If not, mark it as False.'''



               
prompt_map = {
    'False Premise': false_premise_prompt,
    'Unanswerable Questions': unasnwerable_question_prompt,
    'Subject Ambiguity': subject_ambiguity_prompt,
    'Subjective Interpretations': subjective_interpretations_prompt,
    'Unclear User Background': unclear_user_background_prompt,
    'Hidden Human Preferences': hidden_human_preferences_prompt
}


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_inference(question, response, question_type, img_path):
    prompt = prompt_map[question_type]
    prompt_template = prompt + "Here is the question provided to the model: {question}. And here is the model's response you need to evaluate: {response} \nThe format of your answer should be as follows:\nExplanation: Your reasoning for flagging the response as True or False. \n Flag: True/False"
    

    prompt = prompt_template.format(question=question, response=response)
    base64_image = encode_image(img_path)
    # print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[

            
            {
                "role": "user", 
                "content": [
                    {'type': "text", "text": prompt}, 
                    {"type": "image_url", 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
                    ]
            }
            
        ], max_tokens=300,
    )
    return response.choices[0].message.content



argparser = argparse.ArgumentParser()
argparser.add_argument('--method', type=str)
args = argparser.parse_args()


src_data = []
with open('../eval_results/' + args.method+ '_eval_results.jsonl', 'r') as f:
    # add data to src_data
    for line in f:
        src_data.append(json.loads(line.strip()))

for item in tqdm(src_data):
    question, response = item['question'], item['answer']
    img_path = item['img_path']
    question_type = type_map[question]
    count = 0
    while count < 3:
        try:
            result = gpt_inference(question, response, question_type, img_path)
            item['evaluation'] = result
            break
        except:
            count += 1
            print('retrying')
            continue


# write to file
with open('../eval_results/' + args.method + '_flagged.jsonl', 'w') as f:
    for item in src_data:
        f.write(json.dumps(item) + '\n')


#metric calculation
#tier1
false_premise = {'total': 0, 'true': 0}
unanswerable = {'total': 0, 'true': 0}

#tier2
subject_ambiguity = {'total': 0, 'ambiguous': 0, 'true': 0}
subjective_interpretations = {'total': 0, 'ambiguous':0, 'true': 0}
unclear_user_background = {'total': 0, 'ambiguous': 0, 'true': 0}

#tier3
hidden_human_preferences = {'total': 0, 'true': 0}



results = {
    'False Premise': false_premise,
    'Unanswerable Questions': unanswerable,
    'Subject Ambiguity': subject_ambiguity,
    'Subjective Interpretations': subjective_interpretations,
    'Unclear User Background': unclear_user_background,
    'Hidden Human Preferences': hidden_human_preferences
}

  
    
with open('../eval_results/' + args.method+ '_flagged.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        question_type = type_map[data['question']]
        results[question_type]['total'] += 1
        flag = data['evaluation'].split('Flag:')[-1].strip()
        if flag == 'True':
            results[question_type]['true'] += 1
        elif flag == 'Ambiguous':
            results[question_type]['ambiguous'] += 1

# print(results)
            
for key,value in results.items():
    if 'ambiguous' in value.keys():
        rate = {'True': round(value['true']/value['total'],2), 'Ambiguous': round(value['ambiguous']/value['total'],2)}
    else:
        rate = round(value['true']/value['total'],2)
    results[key] = rate
    
print(results)

        
    
    
