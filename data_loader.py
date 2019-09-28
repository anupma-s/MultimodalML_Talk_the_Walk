import json
import os
import argparse


UNK_TOKEN = '__UNK__'
START_TOKEN = '__START__'
END_TOKEN = '__END__'
PAD_TOKEN = '__PAD__'
SPECIALS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

from torch.utils.data.dataset import Dataset



class ActionAwareDictionary:

    def __init__(self):
        self.aware_id2act = ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']
        self.aware_act2id = {v: k for k, v in enumerate(self.aware_id2act)}

    def encode(self, msg):
        if msg in self.aware_act2id:
            return self.aware_act2id[msg]+1
        return -1

    def decode(self, id):
        return self.aware_id2act[id-1]

    def __len__(self):
        return len(self.aware_id2act) + 1



neighborhoods = ['hellskitchen', 'williamsburg', 'eastvillage', 'fidi', 'uppereast']
boundaries = dict()
boundaries['hellskitchen'] = [3, 3]
boundaries['williamsburg'] = [2, 8]
boundaries['eastvillage'] = [3, 4]
boundaries['fidi'] = [2, 3]
boundaries['uppereast'] = [3, 3]

class TalkTheWalk(Dataset):
    """Dataset loading for natural language experiments.

    Only contains trajectories taken by human annotators
    """

    def __init__(self, data_dir, set, last_turns=1, min_freq=3, min_sent_len=2, orientation_aware=False,
                 include_guide_utterances=True):
        self.dialogues = json.load(open(os.path.join(data_dir, 'talkthewalk.{}.json'.format(set))))
        self.act_aware_dict = ActionAwareDictionary()


        self.data = list()

        for config in self.dialogues:
            loc = config['start_location']
            neighborhood = config['neighborhood']
            boundaries = config['boundaries']

            dialogue_context = list()
            for msg in config['dialog']:
                if msg['id'] == 'Tourist':
                    act = msg['text']
                    act_id = self.act_aware_dict.encode(act)
                    if act_id < 0:
                        if len(msg['text'].split(' ')) > min_sent_len:
                            dialogue_context.append((msg['text']))


                else:
                    dialogue_context.append(msg['text'])
            self.data.append(dialogue_context)

    def __getitem__(self, index):
                        return self.data

    def __len__(self):
                        return 100  # len(self.data['target'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')

args = parser.parse_args()
data_dir = args.data_dir

# train_set = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
# valid_set = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
test_set = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))





# train_data = TalkTheWalk(data_dir, 'train')
# valid_data = TalkTheWalk(data_dir, 'train')
test_data = TalkTheWalk(data_dir, 'train')
