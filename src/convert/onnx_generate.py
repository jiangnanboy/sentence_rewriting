
import torch
import os
import onnxruntime
from onnxruntime.transformers.gpt2_helper import Gpt2Helper
import numpy as np
from math import log

from src.core.dataset import GetDataset
from src.core.train import load_tokenizer
from src.core.train import load_config

from src.ner.extract_kw import extract_keywords_textrank, extract_keywords_tfidf


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def get_tokenizer(model_path, SPECIAL_TOKENS):
    return load_tokenizer(model_path, SPECIAL_TOKENS)

def get_example_inputs(tokenizer, sentence, SPECIAL_TOKENS):
    keywords = extract_keywords_textrank(sentence)
    if len(keywords) == 0:
        keywords = extract_keywords_tfidf(sentence)
    keywords = GetDataset.join_keywords(keywords, randomize=False)
    input_text = SPECIAL_TOKENS['bos_token'] + keywords + SPECIAL_TOKENS['sep_token']
    encodings_dict = tokenizer.encode_plus(input_text)

    input_ids = encodings_dict['input_ids']
    input_ids = input_ids[1:len(input_ids) - 1]

    attention_mask = encodings_dict['attention_mask']
    attention_mask = attention_mask[1: len(attention_mask) - 1]
    attention_mask = torch.tensor(attention_mask, dtype=torch.float32)

    position_ids_tensor = (attention_mask.long().cumsum(-1) - 1)
    position_ids_tensor.masked_fill_(position_ids_tensor < 0, 0)
    position_ids_tensor = position_ids_tensor.unsqueeze(0)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
    attention_mask_tensor = attention_mask.unsqueeze(0)

    return input_ids_tensor, attention_mask_tensor, position_ids_tensor, keywords

def get_empty_past(input_ids, num_attention_heads, hidden_size, num_layer):

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(DEVICE))

    return empty_past


def onnx_inference(sentence, config, tokenizer, session, SPECIAL_TOKENS):
    input_ids_tensor, attention_mask_tensor, position_ids_tensor = get_example_inputs(tokenizer, sentence, SPECIAL_TOKENS)
    empty_past = get_empty_past(input_ids_tensor, config.n_head, config.n_embd, config.n_layer)

    print('input_ids_tensor shape'.format(input_ids_tensor))
    print('attention_mask_tensor shape'.format(attention_mask_tensor))
    print('position_ids_tensor shape'.format(position_ids_tensor))
    print('empty_past shape'.format(empty_past))

    ort_input = {'input_ids': np.ascontiguousarray(input_ids_tensor.cpu().numpy()),
                 'attention_mask': np.ascontiguousarray(attention_mask_tensor.cpu().numpy()),
                 'position_ids': np.ascontiguousarray(position_ids_tensor.cpu().numpy())}

    for i, past_i in enumerate(empty_past):
        ort_input[f'past_{i}'] = np.ascontiguousarray(past_i.cpu().numpy())
    ort_outputs = session.run(None, ort_input)

    return ort_outputs


def load_onnx_model(onnx_path):
    print("onnx model load...")
    session = onnxruntime.InferenceSession(onnx_path)
    return session

def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                 past_sequence_length=past[0].size(3),
                                                 sequence_length=input_ids.size(1),
                                                 config=config)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, DEVICE)

    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past,
                                               output_buffers, output_shapes)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes,
                                                            return_numpy=False)
    return outputs

def test_generation(tokenizer, config, input_text, SPECIAL_TOKENS, ort_session=None, num_tokens_to_produce=100):
    eos_token_id = tokenizer.eos_token_id
    input_ids_tensor, attention_mask_tensor, position_ids_tensor, keywords = get_example_inputs(tokenizer, input_text, SPECIAL_TOKENS)
    empty_past = get_empty_past(input_ids_tensor, config.n_head, config.n_embd, config.n_layer)
    batch_size = input_ids_tensor.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids_tensor.clone()

    for step in range(num_tokens_to_produce):

        outputs = inference_with_io_binding(ort_session, config, input_ids_tensor, position_ids_tensor, attention_mask_tensor, empty_past)
        next_token_logits = outputs[0][:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)

        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids_tensor = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(DEVICE)
        position_ids_tensor = (position_ids_tensor[:, -1] + 1).reshape(batch_size, 1)
        attention_mask_tensor = torch.cat([attention_mask_tensor, torch.ones([batch_size, 1]).type_as(attention_mask_tensor)], 1).to(DEVICE)

        empty_past = []
        for i in range(config.n_layer):
            past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], np.ndarray) else outputs[i + 1].clone().detach()
            empty_past.append(past_i.to(DEVICE))

        if torch.all(has_eos):
            break

    for i, output in enumerate(all_token_ids):
        print("----------------------")
        result = tokenizer.decode(output, skip_special_tokens=True)
        text_len = len(','.join(keywords))
        print(result[text_len:])

if __name__ == '__main__':

    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    model_path = os.path.join(path, "model/checkpoint")
    onnx_model_path = os.path.join(path, "model/convert_path/model.onnx")

    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}

    tokenizer = get_tokenizer(model_path, SPECIAL_TOKENS)
    config = load_config(tokenizer, model_path)

    session = load_onnx_model(onnx_model_path)

    sentence = '动员公众参与环保行动,保护绿水青山,普及绿色发展理念。'
    keywords = extract_keywords_textrank(sentence)
    if len(keywords) == 0:
        keywords = extract_keywords_tfidf(sentence)
    print('keywords: {}'.format(keywords))

    # ort_outputs = onnx_inference(sentence, config, tokenizer, session, SPECIAL_TOKENS)
    #
    # print('ort_outputs: {}'.format(ort_outputs.shape))

    test_generation(tokenizer, config, sentence, SPECIAL_TOKENS,session)




