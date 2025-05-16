from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import json
from jinja2 import Template
from tqdm import tqdm
import re
import argparse
from prompt import PROMPT

MAX_RETRY = 10

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
            content.append({"type": "image", "image": part.strip()})

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
    model_path = args.model_path

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 50},
        tensor_parallel_size=args.tensor_parallel_size,
    )
    processor = AutoProcessor.from_pretrained(model_path)

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
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        stop_token_ids = None
        sampling_params = SamplingParams(temperature=0.0,
                                    max_tokens=2048,
                                    stop_token_ids=stop_token_ids)
        image_data, _ = process_vision_info(messages)
        retry = MAX_RETRY  
        while retry > 0:
            try:
                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": image_data
                        },
                    },
                    sampling_params=sampling_params
                )
                break
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
    parse.add_argument('--input_file', type=str, help='The data.jsonl of RAG-IG.', default='./data.jsonl')
    parse.add_argument('--output_dir', type=str, help='The path for generation result. It should be a .jsonl file.', default='./qwen2vl_answer.jsonl')
    parse.add_argument('--tensor_parallel_size', type=int, help='Argument of vllm', default=8)
    parse.add_argument('--model_path', type=str, help='The path of Qwen2VL model.')

    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
