import json
import re
from PIL import Image
import requests
import torch
from rouge_score import rouge_scorer
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import jieba
from rouge import Rouge
import numpy as np
import argparse

def calculate_rouge(candidate, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def edit_distance(listA, listB):
    n = len(listA)
    m = len(listB)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if listA[i - 1] == listB[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + 1  # replace
                )
    
    return 1 - dp[n][m] / max(m,n)

def Kendall_Score(ordered_list, reference_list):
    
    reference_set = set(reference_list)
    index_map = {value: index for index, value in enumerate(reference_list)}
    
    filtered_list = [item for item in ordered_list if item in reference_set]
    if len(ordered_list) == 0 and len(reference_list) > 0 :
        return 0
    if len(reference_list) == 1:
        return len(filtered_list) / len(ordered_list)  
    if len(ordered_list) == 1:
        return len(filtered_list) / len(reference_list)  
    
    num_concordant = 0
    num_discordant = 0

    length = len(filtered_list)
    for i in range(length):
        for j in range(i + 1, length):
            if index_map[filtered_list[i]] < index_map[filtered_list[j]]:
                num_concordant += 1
            else:
                num_discordant += 1
    
    total_pairs = len(ordered_list) * (len(ordered_list)-1) / 2  
    kendall_score = num_concordant / total_pairs
    return kendall_score

def extract_image_context(markdown_text, image_number, context_length=50):
    pattern = re.compile(rf'!\[\]\(IMG#{image_number}\)')
    
    matches = list(pattern.finditer(markdown_text))
    
    contexts = []
    for match in matches:
        start = max(0, match.start() - context_length)
        end = min(len(markdown_text), match.end() + context_length)
        
        context = markdown_text[start:end]
        contexts.append(context)
    
    return contexts

def fix_json(json_string):

    if not json_string.strip().startswith('{'):
        json_string = json_string[json_string.find("{"):]

    if not json_string.strip().endswith('}'):
        json_string += '}'

    if not json_string[:-1].strip().endswith('\"'):
        json_string = json_string[:-1] + '\"' + '}'
    
    prefix, suffix = json_string.split("\"answer\": \"", 1)
    infix, _suffix = suffix.rsplit("\"", 1)
    infix = infix.replace("\"","\'") 

    return "{\"answer\": \"" + infix.replace("\\n", "\n").replace("\n", "<br>")  + "\"" + _suffix

def transform_model_answer(answer):
    try:
        answer = answer.strip().strip("```").strip("json").strip()
        answer = fix_json(answer)
        answer = json.loads(answer)["answer"]
        return answer
    except Exception as e:
        return False
    

def extract_text_and_images(markdown_str):
    context_length = 20

    image_pattern = r'!\[.*?\]\(IMG#(\d+)\)'
    document_pattern = r'<sup>\[.*?\]\(.*?\)</sup>'

    images = re.findall(image_pattern, markdown_str)

    text_without_documents = re.sub(document_pattern, '', markdown_str)
    text_without_images = re.sub(image_pattern, '', text_without_documents)

    cleaned_text = text_without_images.strip()

    image_matches = list(re.finditer(image_pattern, text_without_documents))
    image_contexts = {}
    for match in image_matches:
        image_number = match.group(1)
        start = max(0, match.start() - context_length)
        end = min(len(text_without_documents), match.end() + context_length)
        context = text_without_documents[start:end]
        context = re.sub(image_pattern, '', context)
        image_contexts[image_number] = context

    return cleaned_text, images, image_contexts

def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1, tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity.item()

def load_stopwords(filename):
    """Load a list of stopwords from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

def remove_stopwords(text):
    """Remove stopwords from segmented text."""
    words = jieba.cut(text)
    stopwords = load_stopwords("./score/cn_stopwords.txt")
    cleaned_words = [word for word in words if word not in stopwords]
    return cleaned_words

def main(args):
    input_file = args.input_file
    output_dir = args.output_dir

    datas = []
    with open(input_file, 'r') as fi:
        lines = fi.readlines()
        for line in lines[:]:
            datas.append(json.loads(line))
    
    embedding_model = SentenceTransformer(args.embedding_model_path).cuda()
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).cuda()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    rouge = Rouge()

    fo = open(output_dir, 'w')
    total_rouge1, total_ed, total_kendall, total_align, total_clip = 0.0, 0.0, 0.0, 0.0, 0.0
    num = 0
    format_invalid = 0
    hallucination = 0

    for i, d in tqdm(enumerate(datas[:])):
        try:
            images = d['images']
            img_list = [img for imgs in images for img in imgs]
            model_answer = d['model_answer']
            gt_answer = d['gt_answer']

            model_answer = transform_model_answer(model_answer)
            if model_answer == False:
                format_invalid += 1
                continue
            model_answer_text, model_img_list, model_img_context = extract_text_and_images(model_answer)

            flag = True
            for img_number in model_img_context.keys():
                if int(img_number) > len(img_list):
                    flag = False
                    break
            if flag == False:
                hallucination += 1
                continue

            gt_answer_text, gt_img_list, gt_img_context = extract_text_and_images(gt_answer)


            model_text_seg = ' '.join(remove_stopwords(model_answer_text))
            ref_text_seg = ' '.join(remove_stopwords(gt_answer_text))

            # rouge1_score, rouge2_score, rougel_score = calculate_rouge(model_text_seg, ref_text_seg)
            rouge_score = rouge.get_scores(model_text_seg, ref_text_seg)
            rouge1_score = rouge_score[0]["rouge-1"]['f']

            d['rouge1_score'] = rouge1_score


            ed_score = edit_distance(model_img_list, gt_img_list)
            kendall_score = Kendall_Score(model_img_list, gt_img_list)

            # The image selection difficulty increases with the number of candidates (i.e. len(img_list)) 
            # but decreases with the number of ground truth images (i.e. len(gt_img_list)). 
            # We apply corresponding penalties and rewards to make the Kendall score more discriminative
            beta = 0.5
            kendall_score = (kendall_score * (1 + beta * ((len(img_list) - len(gt_img_list)) / len(img_list)))) / (1 + beta)
            d['edit_distance'] = ed_score
            d['kendall_score'] = kendall_score
            
            
            # image-text consistency
            alignment_score = []
            clip_score = []

            for img_number in model_img_context.keys():
                if img_number not in gt_img_context.keys():
                    continue
                embeddings = embedding_model.encode([model_img_context[img_number], gt_img_context[img_number]])
                sim = embedding_model.similarity(embeddings, embeddings)[0][1].item()
                if sim >= 1.0:
                    sim = 1.0
                elif sim <= 0.7:
                    sim = 0
                else:
                    sim = (sim - 0.7) / 0.3
                alignment_score.append(sim)
            if len(alignment_score) > 0:
                d['alignment_score'] = np.mean(alignment_score)
            else:
                d['alignment_score'] = 0

            # CLIP score
            for img_number in model_img_context.keys():
                img_url = img_list[int(img_number) - 1]
                image = Image.open(requests.get(img_url, stream=True).raw)
                
                inputs = clip_processor(text=model_img_context[img_number][:25], images=image, return_tensors="pt", padding=True).to('cuda')
                outputs = clip_model(**inputs)
                image_emb = outputs.image_embeds[0]
                text_emb = outputs.text_embeds[0]
                sim = cosine_similarity(image_emb, text_emb)
                if sim >= 0.4:
                    sim = 1.0
                elif sim < 0.1:
                    sim = 0
                else:
                    sim = (sim - 0.1) / 0.3
                clip_score.append(sim)
            if len(clip_score) > 0:
                d['clip_score'] = np.mean(clip_score)
            else:
                d['clip_score'] = 0
            

            num += 1
            fo.write(json.dumps(d, ensure_ascii=False) + '\n')
            fo.flush()

            total_rouge1 += d['rouge1_score']
            total_ed += d['edit_distance']
            total_kendall += d['kendall_score']
            total_align += d['alignment_score']
            total_clip += d['clip_score']

        except Exception as e:
            print(e)
            continue


    print(f'----------eval result of {num} valid cases-----------')
    print(f'rouge-1: {total_rouge1 / num}')
    print(f'edit distance: {total_ed / num}')
    print(f'kendall score: {total_kendall / num}')
    print(f'alignment score: {total_align / num}')
    print(f'clip score: {total_clip / num}')
    print(f'integrated_score: {(total_align + total_ed + total_kendall + total_align + total_clip) / 5 / num}')
    print(f'Invalid format num: {format_invalid}')
    print(f'Hallucination num: {hallucination}')
    fo.close()    


def parse_arguments():
    parse = argparse.ArgumentParser(description='Generation setting of gpt4o.')
    parse.add_argument('--input_file', type=str, help='The model answer file.')
    parse.add_argument('--output_dir', type=str, help='The path for evaluation result. It should be a .jsonl file.')
    parse.add_argument('--clip_model_path', type=str, help='The model used to calculate Clip-Score.', default='openai/clip-vit-large-patch14')
    parse.add_argument('--embedding_model_path', type=str, help='The model used to calculate Alignment-Score.',default='TencentBAC/Conan-embedding-v1')

    args = parse.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
