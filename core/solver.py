import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        basic sum attention, simply attention extractor, no entropy,
        no kl, yes selector, yes 1.0, no combined 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-10')
        self.gpu = kwargs.pop('gpu','/gpu:0')
        self.print_every = kwargs.pop('print_every', 100)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer   

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

   

    def train(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features1 = self.data['features1']
        features2 = self.data['features2']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features1 = self.val_data['features1']
        val_features2 = self.val_data['features2']
        n_iters_val = int(np.ceil(float(val_features1.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        with tf.device(self.gpu):
            loss,h1,h2 = self.model.build_model() # initalize model
            tf.get_variable_scope().reuse_variables() # reuse trained variables
            _,_, generated_captions = self.model.build_sampler(max_len=20) 

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
           
        # summary op   
        tf.scalar_summary('batch_loss', loss)
        tf.scalar_summary('h1', h1)
        tf.scalar_summary('h2', h2)
        
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.histogram_summary(var.op.name+'/gradient', grad)
        summary_op = tf.merge_all_summaries() 

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch
        
        # gpu setup (auto search for gpu and grow memory when necessary)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        # map to gpu, start training
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                # permutate data after each epoch
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch1 = features1[image_idxs_batch]
                    
                    features_batch2 = features2[image_idxs_batch]

                    feed_dict = {self.model._features_layer1: features_batch1, 
                                 self.model._features_layer2: features_batch2,
                                 self.model.captions: captions_batch,
                                 self.model.dropout_keep_prob: 0.5}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" %(j+1, gt)  
                        feed_dict_ = {self.model._features_layer1: features_batch1, 
                                      self.model._features_layer2: features_batch2,
                                      self.model.dropout_keep_prob: 1.0}                  
                        gen_caps = sess.run(generated_captions, feed_dict_)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0
                
                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features1.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch1 = val_features1[i*self.batch_size:(i+1)*self.batch_size]
                        
                        features_batch2 = val_features2[i*self.batch_size:(i+1)*self.batch_size]
                       
                        feed_dict = {self.model._features_layer1: features_batch1, 
                                     self.model._features_layer2: features_batch2,
                                     self.model.dropout_keep_prob: 1.0}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                    
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)

                    
            
         
    def visualize_samples(self, data, split='test',num=10):
        '''
        Args:
            - data: dictionary with the following keys:
            - split: 'train', 'val' or 'test'
        '''

        features1 = self.data['features1']
        features2 = self.data['features2']

        # build a graph to sample captions
        with tf.device(self.gpu):
            alphas1,alphas2,sampled_captions = self.model.build_sampler(max_len=20)
        
        # configure gpu
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # map to gpu, start testing
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch1,features_batch2,image_files = sample_coco_minibatch(data, self.batch_size)
            
            feed_dict = {self.model._features_layer1: features_batch1, 
                         self.model._features_layer2: features_batch2,
                         self.model.dropout_keep_prob: 1.0}
            alps1,alps2,sam_cap = sess.run([alphas1,alphas2,sampled_captions],feed_dict) 
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            for n in range(num):
                print "Sampled Caption: %s" %decoded[n]

                img = ndimage.imread(image_files[n])
                plt.subplot(5, 8, 1)
                plt.imshow(img)
                plt.axis('off')

                # Plot images with attention weights 
                words = decoded[n].split(" ")
                idx = 2
                for t in range(len(words)):
                    if t > 18:
                        break
                    plt.subplot(5, 8,idx)
                    plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
                    plt.imshow(img)
                    alp_curr = alps1[n,t,:].reshape(7,7)
                    alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
                    plt.imshow(alp_img, alpha=0.85)
                    plt.axis('off')

                    plt.subplot(5, 8,idx+1)
                    plt.text(0, 1, '%s'%(words[t]) , color='black', backgroundcolor='white', fontsize=8)
                    plt.imshow(img)
                    alp_curr = alps2[n,t,:].reshape(14,14)
                    alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                    plt.imshow(alp_img, alpha=0.85)
                    plt.axis('off')

                    idx += 2

                plt.show()





