import os
from glob import glob
import pandas as pd
import yaml

def main():
    config_file = './dataset_dirs.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # ----- All dataset
    train_file = './dataset_csv/full_datasets/train/all_data_nodir.csv'
    test_file = './dataset_csv/full_datasets/test/all_data_nodir.csv'
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    for q in config['train']:
        k, v = list(q.items())[0]
        df_train.loc[(df_train.datasetName == k) & (df_train.split == 'train'), 'folderDirectory'] = v

    for q in config['val']:
        k, v = list(q.items())[0]
        if k in ['fastmri_brain', 'fastmri_prostate_T2']:
            df_train.loc[(df_train.datasetName == k) & (df_train.split == 'val'), 'folderDirectory'] = v
        else:
            df_test.loc[(df_test.datasetName == k) & (df_test.split == 'val'), 'folderDirectory'] = v

    for q in config['test']:
        k, v = list(q.items())[0]
        df_test.loc[(df_test.datasetName == k) & (df_test.split == 'test'), 'folderDirectory'] = v

    df_train.to_csv('./dataset_csv/full_datasets/train/all_data.csv', index=False)
    df_test.to_csv('./dataset_csv/full_datasets/test/all_data.csv', index=False)
    

    # ----- fastMRI subsets
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/seed_0/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/seed_0/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/seed_1/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/seed_1/', f.split('/')[-1]), index=False)
                
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/seed_2/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/seed_2/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/seed_3/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/seed_3/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/fastmri_sliced/for_debug/train/seed_4/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/train/seed_4/', f.split('/')[-1]), index=False)
                
                
    path_test = './dataset_csv/fastmri_sliced/for_debug/test/*.csv'
    files_test = glob(path_test)

    for f in files_test:
        df = pd.read_csv(f)
        for q in config['val']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/fastmri_sliced/test/', f.split('/')[-1]), index=False)

    # ----- 90% P,  10% Q
    path_train = './dataset_csv/p90_q10/for_debug/seed_0/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/p90_q10/seed_0/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/p90_q10/for_debug/seed_1/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/p90_q10/seed_1/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/p90_q10/for_debug/seed_2/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/p90_q10/seed_2/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/p90_q10/for_debug/seed_3/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/p90_q10/seed_3/', f.split('/')[-1]), index=False)
        
    path_train = './dataset_csv/p90_q10/for_debug/seed_4/*.csv'
    files_train = glob(path_train)

    for f in files_train:
        df = pd.read_csv(f)
        for q in config['train']:
            k, v = list(q.items())[0]
            if k in ['fastmri_brain', 'fastmri_knee']:
                df.loc[df.datasetName == k, 'folderDirectory'] = v
        df.to_csv(os.path.join('./dataset_csv/p90_q10/seed_4/', f.split('/')[-1]), index=False)


if __name__ == '__main__':
    main()