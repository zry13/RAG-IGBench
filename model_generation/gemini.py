import json
import re
import argparse
import os
import base64
import requests
from jinja2 import Template
from tqdm import tqdm
from prompt import PROMPT
import google.generativeai as genai

MAX_RETRY = 100


def url_to_base64(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        image_data = response.content
        base64_data = base64.b64encode(image_data).decode('utf-8')

        return base64_data
    except:
        return False
    
def convert_prompt(prompt):
    # Regular expression to extract parts between <img> </img>
    img_pattern = re.compile(r'<img>(.*?)</img>', re.DOTALL)
    parts = img_pattern.split(prompt)
    
    content = []
    
    # Identify if the sequence starts with text or image, and add them to the list
    for index, part in enumerate(parts):
        if index % 2 == 0:
            # Even index: text part
            if part.strip():  # If there is text present
                content.append({"type": "text", "text": part.strip()})
        else:
            # Odd index: image URL part
            img_url = part.strip()
            retry = 5
            while retry >= 0:
                try:
                    img_data = url_to_base64(img_url)
                    break
                except:
                    retry -= 1
            if retry < 0:
                return False
            content.append({'mime_type':'image/jpeg','data':img_data})

    return content

def get_content(query, documents, images):
    template = Template(PROMPT)
    img_list = []
    num = 0

    for i, imgs in enumerate(images):
        urls = []
        for j, image_url in enumerate(imgs):
            num += 1
            urls.append(f"(IMG#{num}) <img>{image_url}</img>\n")
        img_list.append(urls)
        
    notes = [{'doc':doc, 'img':img} for (doc, img) in zip(documents, img_list)]
    prompt = template.render(query=query, notes=notes)  
    content = convert_prompt(prompt)

    return content

def main(args):
    
    input_file = args.input_file
    output_dir = args.output_dir
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", args.api_key), transport="rest")
    model = genai.GenerativeModel("gemini-1.5-pro")

    datas = [] 
    with open(input_file, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            datas.append(json.loads(line))
    
    error_num = 0
    fo = open(output_dir, 'w')

    for i, d in tqdm(enumerate(datas, start=1)):
        query = d['query']
        documents = d['documents']
        images = d['images']
             
        content = get_content(query, documents, images)

        retry = MAX_RETRY                
        while retry > 0:
            try:
                answer = model.generate_content(content).text
                break
            except:
                print('Generate error.')
                retry -= 1
        if retry <= 0:
            error_num += 1
            print(f'error_num:{error_num}')
            continue
        
        d['model_answer'] = answer

        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        fo.flush()

    fo.close()

def parse_arguments():
    parse = argparse.ArgumentParser(description='Generation setting of gpt4o.')
    parse.add_argument('--input_file', type=str, help='The data.json of RAG-IG.', default='./data.json')
    parse.add_argument('--output_dir', type=str, help='The path for generation result. It should be a .jsonl file.', default='./gemini_answer.jsonl')
    # parse.add_argument('--images_root_path', type=str, help='The root path of images folder.', default='./images')
    parse.add_argument('--api_key', type=str, help='Keys for gemini. If GEMINI_API_KEY is set in the environment variables, you can omit this one.')

    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
