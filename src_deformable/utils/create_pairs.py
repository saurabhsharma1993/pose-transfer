import sys
sys.path.append('../')
import pandas as pd
import pose_transform
import pose_utils
from itertools import permutations
from opts import opts
import re

opt = opts().parse()

def make_pair_nonvid(df):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[0:1]), axis=1)
    df['person'] = persons
    fr, to = [], []
    for person in pd.unique(persons):
        pairs = list(zip(*list(permutations(df[df['person'] == person]['name'], 2))))
        if len(pairs) != 0:
            fr += list(pairs[0])
            to += list(pairs[1])
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df

def make_pairs(df):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[:-1]), axis=1)
    df['person'] = persons
    fr, to = [], []
    for person in pd.unique(persons):
        pair_fr = df[df['person'] == person]
        # print(person,pair_fr.shape[0])
        num_rows = pair_fr.shape[0]
        i = 0
        for index, row in pair_fr.iterrows():
            if(i+2<num_rows):
                fr += [row['name']]
                to += [pair_fr.iloc[i+2]['name']]
            i += 1
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df

def make_pairs_iterative(df):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[:-1]), axis=1)
    df['person'] = persons

    seq = [[] for i in range(opts.frame_diff+1)]
    for person in pd.unique(persons):

        # skipping certain action classes . . comment below lines to take full dataset
        # m = re.search(r'act_([0-9]{2})', person)
        # act_id = int(m.groups()[0])
        # if (act_id < 14):
        #     continue


        pair_fr = df[df['person'] == person]
        # print(person,pair_fr.shape[0])
        num_rows = pair_fr.shape[0]
        i = 0
        for index, row in pair_fr.iterrows():
            if(i%10!=0):
                i +=1
                continue
            if(i+2*opts.frame_diff <num_rows):
                seq[0] += [row['name']]
                # change for variable difference in time steps in video
                for j in range(1,opts.frame_diff+1):
                    # adding ten frames for every step in sequence
                    seq[j] += [pair_fr.iloc[i+j*2]['name']]
            i += 1
    pair_df = pd.DataFrame(index=range(len(seq[0])))
    for j in range(opts.frame_diff + 1):
        pair_df['seq' + str(j)] = seq[j]
    return pair_df

def make_pairs_restricted(df):
    persons = df.apply(lambda x: '_'.join(x['name'].split('_')[:-1]), axis=1)
    df['person'] = persons
    fr, to = [], []
    for person in pd.unique(persons):

        # take only action classes from walking action types
        m = re.search(r'act_([0-9]{2})', person)
        act_id = int(m.groups()[0])
        if (act_id < 14):
            continue

        pair_fr = df[df['person'] == person]
        # print(person,pair_fr.shape[0])
        num_rows = pair_fr.shape[0]
        i = 0
        for index, row in pair_fr.iterrows():
            if(i+2<num_rows):
                fr += [row['name']]
                to += [pair_fr.iloc[i+2]['name']]
            i += 1
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df

if __name__ == "__main__":
    df_keypoints = pd.read_csv(opt.annotations_file_train, sep=':')
    # df = filter_not_valid(df_keypoints)
    df = df_keypoints
    print ('Compute pair dataset for train...')
    if(opt.pose_dim==16):
        pairs_df_train = make_pairs(df)
    else:
        pairs_df_train = make_pair_nonvid(df)
    pairs_df_train = pairs_df_train.sample(n=min(opt.images_for_train, pairs_df_train.shape[0]), replace=False,                                                                            random_state=0)
    print ('Number of pairs: %s' % len(pairs_df_train))
    pairs_df_train.to_csv(opt.pairs_file_train, index=False)

    print ('Compute pair dataset for test...')
    df_keypoints = pd.read_csv(opt.annotations_file_test, sep=':')
    # df = filter_not_valid(df_keypoints)
    df = df_keypoints
    if (opt.pose_dim == 16):
        pairs_df_test = make_pairs(df)
    else:
        pairs_df_test = make_pair_nonvid(df)
    pairs_df_test = pairs_df_test.sample(n=min(opt.images_for_test, pairs_df_test.shape[0]), replace=False, random_state=0)
    print ('Number of pairs: %s' % len(pairs_df_test))
    pairs_df_test.to_csv(opt.pairs_file_test, index=False)
