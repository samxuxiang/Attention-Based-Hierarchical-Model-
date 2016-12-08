from scipy import ndimage
from collections import Counter
from core.vgg16 import vgg16
from core.utils import *
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import pdb

def _process_caption_data(caption_file, image_dir, max_length):
    """
    read in caption json file
    add image file path to caption_data
    tokenize caption words 
    delete long sentences
    """
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename} 
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and use 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)

    # delete caption id
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data


def _build_vocab(annotations, threshold=1):
    """ 
    read in all words and build word to id dict
    """
    counter = Counter()
    
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
    # filter out less frequently used words
    vocab = [word for word in counter if counter[word] >= threshold]
    # add start/end/null id
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 2
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    """
    convert caption words to mapping id (int)
    add start/end/null to those shorter sentences
    """
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        # add int id for start of a sentence
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            # add int id for known words
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])

        # add int id for end of a sentence
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    return captions


def _build_file_names(annotations):
    """
    simplify image_id to index
    """
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1
    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    batch_size = 16
    # max sentence length
    max_length = 15
    # minimum word frequency
    word_count_threshold = 2
    vgg_model_path = './data/vgg16_weights.npz'
    """
    # captions for train dataset 
    train_dataset = _process_caption_data(caption_file='data/annotations/captions_train2014.json',
                                          image_dir='image/train2014_resized/',
                                          max_length=max_length)
    # captions for val/test dataset
    val_dataset = _process_caption_data(caption_file='data/annotations/captions_val2014.json',
                                        image_dir='image/val2014_resized/',
                                        max_length=max_length)
    
    # get train/val/test dataset 
    train_cutoff = int(0.1*len(train_dataset))
    val_cutoff = int(0.1 * len(val_dataset)) 
    test_cutoff = int(0.2 * len(val_dataset))
    save_pickle(train_dataset[:train_cutoff], 'data/train/train.annotations.pkl')
    save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    for split in ['train', 'val', 'test']:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))

        # used training set to build the word to id dictionary
        if split == 'train':
            word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)

        # convert caption words to word id (int) with added start/end/null
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        # convert all image_id to int id starting from 0 
        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        # convert caption image_id to int id
        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        # append all captions of a single image to one list
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print "Finished building %s caption dataset" %split

    """
    # extract vgg16 feature maps
    with tf.device('/gpu:0'):
      vggnet = vgg16()
     
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        vggnet.build(vgg_model_path,sess)
        for split in ['train','val','test']:
            file_path = './data/%s/%s.file.names.pkl' % (split, split)
            save_path = './data/%s/%s.pool5.features.hkl' % (split, split)
            img_file_path = load_pickle(file_path)
            n_examples = len(img_file_path)
            print n_examples
            all_feats = np.ndarray([n_examples,7,7, 512], dtype=np.float32)
           
            for start, end in zip(range(0, n_examples, batch_size),
                                range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = img_file_path[start:end]
        
                image_batch = np.array(map(lambda x: imread(x, mode='RGB'), image_batch_file))
                feats = sess.run(vggnet.pool5,feed_dict={vggnet.imgs: image_batch})
                all_feats[start:end,:] = feats
                
                
                print np.sum(all_feats[start:end,:])
                
                print ("Processed %d %s features" % (end, split))

            print n_examples
            # normalize feature vectors
            all_feats = np.reshape(all_feats, [-1, 512])
            mean = np.mean(all_feats, 0)
            var = np.var(all_feats, 0)
            all_feats = (all_feats - mean) / np.sqrt(var)
            all_feats = np.reshape(all_feats, [-1, 7,7, 512])
            
            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))




if __name__ == "__main__":
    main()
