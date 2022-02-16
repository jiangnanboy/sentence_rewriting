import os
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import argparse

from src.sentence_rewriting.dataset import GetDataset
from src.sentence_rewriting.train import load_tokenizer
from src.ner.extract_kw import extract_keywords_tfidf, extract_keywords_textrank

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenerateText():
    def __init__(self, model_path, SPECIAL_TOKENS):
        self.tokenizer = load_tokenizer(model_path)
        self.model = self.load_pretrained_model(self.tokenizer, model_path, SPECIAL_TOKENS)
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def load_pretrained_model(self, tokenizer, load_model_path, SPECIAL_TOKENS=None):
        '''
        load pretrained model
        :param tokenizer:
        :param load_model_path:
        :param SPECIAL_TOKENS:
        :return:
        '''
        print("pretrained model loadding...")
        if SPECIAL_TOKENS:
            gpt2Config = GPT2Config.from_pretrained(load_model_path,
                                                    bos_token_id=tokenizer.bos_token_id,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    sep_token_id=tokenizer.sep_token_id,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    output_hidden_states=False)
        else:
            gpt2Config = GPT2Config.from_pretrained(load_model_path,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    output_hidden_states=False)

        model = GPT2LMHeadModel.from_pretrained(load_model_path, config=gpt2Config)

        if SPECIAL_TOKENS:
            # add special token, model embedding size need to adjust
            model.resize_token_embeddings(len(tokenizer))

        if load_model_path:
            # load model
            model.load_state_dict(torch.load(os.path.join(load_model_path, 'pytorch_model.bin')))

        return model

    def generate_text(self, keywords, args, BEAM_SEARCH=False):
        '''
        generate text
        :param keywords:
        :param args:
        :param BEAM_SEARCH:
        :return:
        '''
        keywords = GetDataset.join_keywords(keywords, randomize=False)
        input_text = SPECIAL_TOKENS['bos_token'] + keywords + SPECIAL_TOKENS['sep_token']
        input_text = self.tokenizer.encode(input_text)
        input_text = input_text[1: len(input_text) - 1] # 去除两端[CLS]与[SEP]
        input_text_encoder = torch.tensor(input_text).unsqueeze(0)
        input_text_encoder = input_text_encoder.to(DEVICE)

        if BEAM_SEARCH:
            # beam-search
            output_text = self.model.generate(input_text_encoder,
                                         do_sample=args.do_sample,
                                         max_length=args.max_len,
                                         num_beams=args.num_beams,
                                         repetition_penalty=args.repetition_penalty,
                                         early_stopping=args.early_stopping,
                                         num_return_sequences=args.num_return_sequences)
        else:
            output_text = self.model.generate(input_text_encoder,
                                         do_sample=args.do_sample,
                                         min_length=args.min_len,
                                         max_length=args.max_len,
                                         top_k=args.top_k,
                                         top_p=args.top_p,
                                         temperature=args.temperature,
                                         repetition_penalty=args.repetition_penalty,
                                         num_return_sequences=args.num_return_sequences)
        result = []
        for i, output in enumerate(output_text):
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            text_len = len(','.join(keywords))
            result.append(text[text_len:])

        return result

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    model_path = os.path.join(path, "model/keywords_generation_model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", default=500, type=int, required=False, help="Maximum generate text length!")
    parser.add_argument('--min_len', default=50, type=int, required=False, help='Minimum generate text length!')
    parser.add_argument('--num_return_sequences', default=10, type=int, required=False, help='The number of return generated sentences!')
    parser.add_argument('--do_sample', default=True, type=bool, required=False, help='Do sample!')
    parser.add_argument('--top_k', default=30, type=int, required=False, help='Top K!')
    parser.add_argument('--top_p', default=0.7, type=float, required=False, help='Top p!')
    parser.add_argument('--temperature', default=0.7, type=float, required=False, help='Temperature!')
    parser.add_argument('--repetition_penalty', default=2.0, type=float, required=False, help='Repetition penalty!')
    parser.add_argument('--num_beams', default=5, type=int, required=False, help='The number of beam search!')
    parser.add_argument('--early_stopping', default=True, type=bool, required=False, help='Early stop for beam search!')
    args = parser.parse_args()

    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}
    generateText = GenerateText(model_path, SPECIAL_TOKENS)

    while True:
        sentence = input("input sentence: ")
        if sentence == 'exit':
            break
        keywords = extract_keywords_textrank(sentence)
        result = generateText.generate_text(keywords, args, True)
        # print generate result
        for index, text in enumerate(result):
            print('{}: {}\n'.format(index + 1, text))



