import json
import argparse

def main(args):
    input_file = args.input_file
    datas = []
    with open(input_file, 'r') as fi:
        lines = fi.readlines()
        for line in lines[:]:
            datas.append(json.loads(line))
    total_rouge1, total_ed, total_kendall, total_align, total_clip = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, d in enumerate(datas):
        total_rouge1 += d['rouge1_score']
        total_ed += d['edit_distance']
        total_kendall += d['kendall_score']
        total_align += d['alignment_score']
        total_clip += d['clip_score']
    num = len(datas)
    print(f'----------eval result of {num} valid cases-----------')
    print(f'rouge-1: {total_rouge1 / num}')
    print(f'edit distance: {total_ed / num}')
    print(f'kendall score: {total_kendall / num}')
    print(f'alignment score: {total_align / num}')
    print(f'clip score: {total_clip / num}')
    print(f'integrated_score: {(total_align + total_ed + total_kendall + total_align + total_clip) / 5 / num}')

def parse_arguments():
    parse = argparse.ArgumentParser(description='Generation setting of gpt4o.')
    parse.add_argument('--input_file', type=str, help='The evaluation file.')

    args = parse.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
