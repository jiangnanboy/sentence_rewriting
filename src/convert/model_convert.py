import torch
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from transformers import GPT2Config

import os
from src.core.train import load_tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pretrained_model(tokenizer, load_model_path, SPECIAL_TOKENS=None):
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
        '''
         do_sample=self.args.do_sample,
         max_length=self.args.max_len,
         num_beams=self.args.num_beams,
         repetition_penalty=self.args.repetition_penalty,
         early_stopping=self.args.early_stopping,
         num_return_sequences=num_return_sequences
         
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
        '''
    else:
        gpt2Config = GPT2Config.from_pretrained(load_model_path,
                                                pad_token_id=tokenizer.pad_token_id,
                                                output_hidden_states=False)

    model = MyGPT2LMHeadModel.from_pretrained(load_model_path, config=gpt2Config)

    if SPECIAL_TOKENS:
        # add special token, model embedding size need to adjust
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        # load model
        model.load_state_dict(torch.load(os.path.join(load_model_path, 'pytorch_model.bin'), map_location=DEVICE))

    return model

def convert_model(model_path, convert_model_path):
    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}
    tokenizer = load_tokenizer(model_path, SPECIAL_TOKENS)
    model = load_pretrained_model(tokenizer, model_path, SPECIAL_TOKENS)
    model.eval()
    print(model.config)
    print("num_attention_heads: {}".format(model.config.n_head))
    print("hidden_size: {}".format(model.config.n_embd))
    print("num_layer: {}".format(model.config.n_layer))

    Gpt2Helper.export_onnx(model, DEVICE, convert_model_path)

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    model_path = os.path.join(path, "model/checkpoint")
    onnx_model_path = os.path.join(path, "model/convert_path/model.onnx")

    convert_model(model_path, onnx_model_path)




