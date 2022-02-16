from torch.utils.data import Dataset, DataLoader
import torch
import random

class GetDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, SPECIAL_TOKENS, randomize=True):
        '''
        build dataset
        :param df:
        :param tokenizer:
        :param max_length:
        :param SPECIAL_TOKENS:
        :param randomize:
        '''
        self.df = df
        self.data_size = len(df)
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.randomize = randomize
        self.max_length = max_length

    @staticmethod
    def join_keywords(keywords, randomize=True):
        keywords_len = len(keywords)
        # sampling and shuffle
        if randomize:
            random_size = random.choice(range(keywords_len + 1))
            keywords = keywords[:random_size]
            random.shuffle(keywords)
        return ','.join(keywords)

    def __getitem__(self, idx):
        keywords, text = self.df.iloc[idx, :].values
        keywords = keywords.split()
        keywords = self.join_keywords(keywords, self.randomize)
        input = self.SPECIAL_TOKENS['bos_token'] + keywords + \
                self.SPECIAL_TOKENS['sep_token'] + text + self.SPECIAL_TOKENS['eos_token']
        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding='max_length')

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        input_ids = input_ids[1: len(input_ids) - 1] # 去除两端[CLS]与[SEP]
        attention_mask = attention_mask[1: len(attention_mask) - 1]

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

    def __len__(self):
        return self.data_size

def get_train_val_dataloader(batch_size, trainset, train_ratio):
    '''
    split trainset to train and val
    :param batch_size:
    :param trainset:
    :param train_ratio
    :return:
    '''

    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    valloader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False)

    return trainloader, valloader, train_dataset, val_dataset
