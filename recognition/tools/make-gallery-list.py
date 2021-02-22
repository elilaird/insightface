import pandas as pd
import requests as re
import numpy as np
import json
import argparse
import os
import sys
from pprint import pprint
from progressbar import ProgressBar, SimpleProgress
from multiprocessing import Pool, Lock
import logging
import itertools


lock = Lock()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Read image data from remote data source and generate data list file'
    )
    parser.add_argument('-r', '--remote', help='remote data endpoint', default=None)
    parser.add_argument('-o', '--output', help='output list file name', required=True)
    parser.add_argument('-e','--encoding', help='specify the encoding of the images.',
                        type=str,
                        default='.jpg',
                        choices=['.jpg', '.png'])
    parser.add_argument('-d', '--datadir', help='path to image directory', default=None)
    parser.add_argument('-c','--csv', help='csv to create list file',default=None)
    parser.add_argument('-m','--metadata', help='url for metadata file', default=None)
    parser.add_argument('-t', '--threading', help='use multiprocessing [0:no, 1:yes]', default=0, choices=[0,1])

    args = parser.parse_args()
    return args

def read_url(rmt_str):
    try:
        with re.get(rmt_str) as r:
            data = r.json()
    except Exception as e:
        print('Failed reading data with error: ', e)
        return None

    return data

def split_key_val_pairs(key_pairs):
    d = {}
    for p in key_pairs:
        k,v = p.split('=')
        d = dict(d, **{k:v})
    return d

def json_2_dataframe(json_list, images=True):
    if images:
        if len(json_list) > 0:
            df = pd.concat([
                pd.DataFrame.from_dict(
                    dict(entry,  **split_key_val_pairs(entry['key_value_pairs'])),
                    orient='columns'
                )
                for entry in json_list
            ], ignore_index=True)
            df = df.loc[:, df.columns != 'key_value_pairs']
        else:
            return None
    else:
        if len(json_list) > 0:
            df = pd.concat([
                pd.DataFrame(
                    entry,
                    index=[idx]
                )
                for idx,entry in enumerate(json_list)
            ], ignore_index=False)

        else:
            return None

    return df

def get_demographics(meta_data):
    df = meta_data[['SubjectID','Gender','Race']].copy()
    conds_g = [
        df['Gender']=='Male',
        df['Gender']=='Female'
    ]
    choices_g = ['M','F']
    conds_r = [
        df['Race']=='Black or African-American',
        df['Race']=='Black or African American',
        df['Race']=='White',
        True #If anything other than ^, then 'Other'
    ]
    choices_r = ['B','B','W','O']
    df['Gender'] = np.select(conds_g, choices_g)
    df['Race'] = np.select(conds_r, choices_r)
    df['Demographic'] = df['Race'] + df['Gender']

    return df[['SubjectID','Demographic']].copy()

def csv_to_list_file(df, fname):
    LOGGER.info("Creating data list: ".format(fname))
    with open(fname, 'w') as f:
        for idx, row in df.iterrows():
            f.write('{}\t{}\t{}\t{}\n'.format(row['imageId'],row['subject'],row['dem'],row['url']))

def df_to_list_file(df, fname):
    LOGGER.info("Creating data list: {} ".format(fname))
    pbar = ProgressBar(widgets=[SimpleProgress()], maxval=df.shape[0]).start()
    with open(fname, 'w') as f:
        for idx, row in df.iterrows():
            f.write('{}\t{}\t{}\t{}\n'.format(row['imageId'],row['subject'],row['dem'],row['dir']))
            pbar.update(value=idx)
    pbar.finish()

def download_worker(args):
    file_list, dir, id_url = args
    id, url = id_url
    file = url.split('/')[-1]
    fname = os.path.join(dir, file)
    if not os.path.isfile(fname):
        with open(fname, 'wb') as f:
            try:
                img_bytes = re.get(url).content
                f.write(img_bytes)
                LOGGER.info("Downloaded image {}:{}".format(id, url))
            except Exception as e:
                LOGGER.error("Failed to download image {}:{} with error {}".format(id, url, e))
    else:
        LOGGER.info("Image {}:{} exists".format(id, fname))

    return (id,fname)

def download_imgs(df, dir):
    LOGGER.info("Downloading images")
    file_dict = {"imageId":[], "dir":[]}
    with Pool(processes=20) as pool:
        LOGGER.info("Starting image download pool")

        params = itertools.product([file_dict], [dir], zip(df['imageId'],df['url'].tolist()))
        pbar = ProgressBar(widgets=[SimpleProgress()], maxval=df.shape[0]).start()

        for idx, row in enumerate(pool.imap(download_worker, params)):
            with lock:
                file_dict["imageId"].append(row[0])
                file_dict["dir"].append(row[1])
            pbar.update(value=idx)
        pbar.finish()

    return pd.concat([df.set_index('imageId'),pd.DataFrame.from_dict(file_dict).set_index('imageId')], axis=1, join='inner').reset_index()

def setup_logger(name):
    '''set up logger'''
    logging.basicConfig(filename='../logs/make-gallery-list.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)

    #create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    #add the handlers to logger
    logger.addHandler(handler)

    return logger



if __name__ == '__main__':

    LOGGER = setup_logger('make-gallery-list')

    args = parse_args()


    remote = args.remote
    ofname = args.output
    encoding = args.encoding
    metaurl = args.metadata
    csv_file = args.csv
    img_dir = args.datadir



    if csv_file is not None:
        df = pd.read_csv(csv_file)
        if img_dir is not None:
            df = download_imgs(df, img_dir)
        df_to_list_file(df, ofname)
        LOGGER.info("List file {} created ".format(ofname))
    elif remote is not None:
        data = read_url(remote)
        metadata = read_url(metaurl)

        if data is not None:
            df_imgs = json_2_dataframe(data)

            if metadata is not None:
                df_meta = json_2_dataframe(metadata, False)
                df_demographics = get_demographics(df_meta)
                pprint(df_demographics.head())
        else:
            LOGGER.error("Failed to create list file from remote data")
            exit()

        # merge and create data list file
        df_imgs['subject'] = df_imgs['subject'].astype(str)
        df_demographics['SubjectID'] = df_demographics['SubjectID'].astype(str)
        df_imgs = df_imgs.rename(columns={"subject": "SubjectID"})

        df_data = pd.concat([df_imgs.set_index('SubjectID'), df_demographics.set_index('SubjectID')], axis=1,
                            join='inner').reset_index()













