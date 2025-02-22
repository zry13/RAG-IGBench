from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from vllm.multimodal.utils import fetch_image
import json
from jinja2 import Template
from tqdm import tqdm
import re
import argparse
from prompt import PROMPT

MAX_RETRY = 10

def get_content(query, documents, images):
    template = Template(PROMPT)
    
    img_list = []
    num = 0

    for i, note in enumerate(images):
        urls = []
        for j, image_url in enumerate(note['imgList']):
            num += 1
            urls.append(f"(IMG#{num}) <image>\n")
        img_list.append(urls)

    notes = [{'doc':doc, 'img':img} for (doc, img) in zip(documents, img_list)]
    content = template.render(query=query, notes=notes)  

    return content

def main():
    input_file = args.input_file
    output_dir = args.output_dir
    model_path = args.model_path
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 50},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=args.tensor_parallel_size
    )

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
        messages = [{'role': 'user', 'content': content}]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(messages,
                                        tokenize=False,
                                        add_generation_prompt=True)

        stop_token_ids = None

        sampling_params = SamplingParams(temperature=0.0,
                                    max_tokens=2048,
                                    stop_token_ids=stop_token_ids)
        retry = MAX_RETRY  
        while retry > 0:
            try:
                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": [fetch_image(url) for imgs in images for url in imgs]
                        },
                    },
                    sampling_params=sampling_params
                )
            except Exception as e:
                print(e)
                retry -= 1
        if retry <= 0:
            error_num += 1
            print(f'error_num:{error_num}')
            continue
        answer = outputs[0].outputs[0].text
        d['model_answer'] = answer
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        fo.flush()

    fo.close()

def parse_arguments():
    parse = argparse.ArgumentParser(description='Generation setting of gpt4o.')
    parse.add_argument('--input_file', type=str, help='The data.json of RAG-IG.', default='./data.json')
    parse.add_argument('--output_dir', type=str, help='The path for generation result. It should be a .jsonl file.', default='./llava_ov_answer.jsonl')
    parse.add_argument('--tensor_parallel_size', type=int, help='Argument of vllm', default=8)
    parse.add_argument('--model_path', type=str, help='The path of Llava-OV / NVLM_D model.')

    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
