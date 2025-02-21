import json
from tqdm import tqdm
import requests
import os
import argparse

def download_image(url, save_path):
    # 发送HTTP GET请求获取图片内容
    response = requests.get(url)
    
    retry = 10
    while retry > 0:
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            retry -= 1
    return False

def main(args):
    input_file = args.input_file
    images_root_path = args.images_root_path

    datas = []
    with open(input_file, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            datas.append(json.loads(line))

    error_num = 0

    for i, d in tqdm(enumerate(datas, start=1)):
        images = d['images']
        img_counter = 0
        for j, imgs in enumerate(images, start=1):
            for img_url in imgs:
                img_counter += 1
                saved_path = os.path.join(images_root_path, i, j, f'IMG#{img_counter}.jpg')
                flag = download_image(img_url.replace('w/720', 'w/540').replace('w/1080', 'w/540'), saved_path)
                if not flag:
                    error_num += 1
                    print(f"Download error {error_num}: id:{i}, DOC#{j}, IMG#{img_counter}, img_url:{img_url}")
    
    return True
            
def parse_arguments():
    parse = argparse.ArgumentParser(description='Generation setting of gpt4o.')
    parse.add_argument('--input_file', type=str, help='The data.json of RAG-IG.', default='./data.json')
    parse.add_argument('--images_root_path', type=str, help='The root path to download images .', default='./images')
    
    args = parse.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)