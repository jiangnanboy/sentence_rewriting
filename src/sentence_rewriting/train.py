from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config, TrainingArguments, Trainer
import torch
import os
import argparse
import random
import numpy as np

from src.util import read_data, split_data

from src.sentence_rewriting.dataset import GetDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    set seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_tokenizer(tokenizer_path, special_token_path=None):
    '''
    load tokenizer
    :param tokenizer_path:
    :param special_token_path:
    :return:
    '''
    print('tokenizer loadding...')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    if special_token_path:
        tokenizer.add_special_tokens(special_token_path)
        print('special tokens added!')
    return tokenizer

def load_pretrained_model(tokenizer, pretrained_model_path, special_token_path=None):
    '''
    model with pretrained model
    :param tokenizer:
    :param pretrained_model_path:
    :param special_token_path:
    :return:
    '''
    print("pretrained model loadding...")
    gpt2Config = GPT2Config.from_pretrained(pretrained_model_path,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False) # Whether or not the model should return all hidden-states
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_path, config=gpt2Config)

    if special_token_path:
        # add special token,model embedding size need to adjust
        model.resize_token_embeddings(len(tokenizer))

    '''
    # freeze selective layers
    # freeze bias and layernorm.weight
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    '''

    # 1.freeze all layers of pretrained model except last n
    for param in model.parameters():
        param.requires_grad = False

    # 2.only the gradients of the last six blocks are trained
    for i, m in enumerate(model.transformer.h):
        if (i + 1) > 6:
            for param in m.parameters():
                param.requires_grad=True

    for param in model.transformer.ln_f.parameters(): # layernorm
        param.requires_grad = True

    for param in model.lm_head.parameters(): # linear
        param.requires_grad=True

    return model.to(DEVICE)

def build_model(tokenizer, model_config, special_token_path=None):
    '''
    model without pretrained model
    :param tokenizer:
    :param model_config:
    :param special_token_path:
    :return:
    '''
    gpt2Config = GPT2Config.from_json_file(model_config)
    model = GPT2LMHeadModel(config=gpt2Config)

    if special_token_path:
        model.resize_token_embeddings(len(tokenizer))
    return model.to(DEVICE)

def train_val(model, tokenizer, train_dataset, val_dataset, param_args):
    '''
    train and val
    :param model:
    :param tokenizer:
    :param train_dataset
    :param val_dataset
    :param param_args
    :return:
    '''
    training_args = TrainingArguments(output_dir=param_args.output_dir,
                                      num_train_epochs=param_args.epochs,
                                      per_device_train_batch_size=param_args.batch_size,
                                      per_device_eval_batch_size=param_args.batch_size,
                                      gradient_accumulation_steps=param_args.gradient_accumulation_steps,
                                      evaluation_strategy=param_args.evaluation_strategy,
                                      fp16=param_args.fp16,
                                      fp16_opt_level=param_args.apex_opt_level,
                                      warmup_steps=param_args.warmup_steps,
                                      learning_rate=param_args.lr,
                                      adam_epsilon=param_args.adam_eps,
                                      weight_decay=param_args.weight_decay,
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      prediction_loss_only=args.prediction_loss_only,
                                      logging_dir=param_args.logging_dir # tensorboard
                                      )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset, # dataset, not dataloader
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer)
    trainer.train()

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    print("Base path : {}".format(path))
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--config_path",
        default=os.path.join(path, 'model/pretrained_model/config.json'),
        type=str,
        required=False,
        help="The path of configration!",
    )
    parser.add_argument(
        '--special_token_path',
        default=os.path.join(path, 'model/pretrained_model/special_tokens_map.json'),
        type=str,
        required=False,
        help='The path of special tokens!'
    )
    parser.add_argument(
        "--vocab_path",
        default=os.path.join(path, 'model/pretrained_model/vocab.txt'),
        type=str,
        required=False,
        help="The path of vocabulary!",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(path, 'data/aiwriter.csv'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=8, type=int, required=False, help="Batch size!"
    )
    parser.add_argument("--lr", default=1.5e-3, type=float, required=False, help="Learning rate!")
    parser.add_argument("--warmup_steps", default=1e2, type=float, required=False, help="LR updated patience coefficient!")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int, required=False, help="How many times to update gradients!")
    parser.add_argument("--weight_decay", default=1e-3, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=500, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument(
        "--train_ratio", default=0.9, type=float, required=False, help="Training set ratio!"
    )
    parser.add_argument(
        "--print_loss", default=1, type=int, required=False, help="How many steps print a loss!"
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(path, 'model/keywords_generation_model'), type=str, required=False, help="The path of model output!"
    )
    parser.add_argument('--prediction_loss_only', default=True, type=bool, required=False, help='When performing evaluation and generating predictions, only returns the loss!')
    parser.add_argument("--logging_dir", default=os.path.join(path, 'model/keywords_generation_model/logs'), type=str, required=False, help="The path of log output (tensorboard)!")
    parser.add_argument(
        "--seed", default=2021, type=int, required=False, help="Python hash seed!"
    )
    parser.add_argument(
        "--use_apex", default=True, type=bool, required=False, help="Use APEX!"
    )
    parser.add_argument("--fp16", default=True, type=bool, required=False, help="APEX fp16!")
    parser.add_argument("--evaluation_strategy", default="epoch", type=str, required=False, help="Evaluation strategy!")
    parser.add_argument("--adam_eps", default=1e-8, type=float, required=False, help="Adam eps,prevent dividing by zero!")
    parser.add_argument("--apex_opt_level", default="o1", type=str, required=False, help="APEX opt level!")

    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model_path
    config_path = args.config_path
    vocab_path = args.vocab_path
    data_path = args.data_path
    special_token_path = args.special_token_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    max_length = args.max_length
    train_ratio = args.train_ratio
    print_loss = args.print_loss
    output_dir = args.output_dir
    logging_dir = args.logging_dir
    seed = args.seed
    use_apex = args.use_apex
    apex_opt_level = args.apex_opt_level
    warmup_steps = args.warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    fp16 =  args.fp16
    evaluation_strategy = args.evaluation_strategy
    
    SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]",
     "bos_token": "[BOS]", "eos_token": "[EOS]"}

    # train data format
    columns = [
        'keywords',
        'content']

    # read data
    pd_data = read_data(data_path, columns)

    # split train and val
    train_set, val_set = split_data(pd_data, train_ratio)

    # load tokenize
    tokenizer = load_tokenizer(pretrained_model_path, SPECIAL_TOKENS)

    # build training and validation set
    trainset = GetDataset(train_set, tokenizer, max_length, SPECIAL_TOKENS)
    valset = GetDataset(val_set, tokenizer, max_length, SPECIAL_TOKENS, randomize=False)

    # build dataload and dataset
    # trainloader, valloader, train_dataset, val_dataset= get_train_val_dataloader(batch_size, trainset, train_ratio)

    # build model with pretrained model
    model = load_pretrained_model(tokenizer, pretrained_model_path, SPECIAL_TOKENS)

    # build model without pretrained model
    # model = build_mode(tokenizer, config_path, SPECIAL_TOKENS)

    # train and val
    train_val(model, tokenizer, trainset, valset, args)
    
