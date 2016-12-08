# =========================================================================================
# Xiang Xu (2016-12-1) code for 10-807 final project
# reuse some of the code from (https://github.com/yunjey/show-attend-and-tell-tensorflow)
#                             (https://github.com/jazzsaxmafia/video_to_sequence)
# other group members: Markus Woodsons, Yubo Zhang
# =========================================================================================

from __future__ import division

import tensorflow as tf

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_layer1=[7,512],dim_layer2=[14,512],
                  dim_embed=512, dim_hidden=1024, n_time_step=16, 
                  alpha_c=[0.0,0.0],alpha_e=0.0):
        """
        basic sum attention, simply attention extractor, yes entropy,
        no kl, no selector, yes 1.0, yes combined 
        """
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.alpha_c = alpha_c
        self.alpha_e = alpha_e
        self.V = len(word_to_idx)
        self.S1 = dim_layer1[0]
        self.D1 = dim_layer1[1]
        self.S2 = dim_layer2[0]
        self.D2 = dim_layer2[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.const_initializer = tf.constant_initializer(0.0)
        self.constant_one = tf.constant_initializer(1.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.small_initializer = tf.constant_initializer(0.01)
        # Place holder for features and captions
        self._features_layer1 = tf.placeholder(tf.float32, [None,self.S1,self.S1,self.D1])
        self._features_layer2 = tf.placeholder(tf.float32, [None,self.S2,self.S2,self.D2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.features_layer1 = tf.reshape(self._features_layer1, [-1, self.S1**2, self.D1])
        self.features_layer2 = tf.reshape(self._features_layer2, [-1, self.S2**2, self.D2])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.e =1e-12

    def _get_initial_lstm(self,name,dim_f,dim_h,features):
        with tf.variable_scope(name):
            features_mean = tf.reduce_mean(features, 1)  # (N,dim_f)

            w_h = tf.get_variable('w_h', [dim_f, dim_h], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [dim_h], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)  # (N,dim_h)

            w_c = tf.get_variable('w_c', [dim_f, dim_h], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [dim_h], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)  # (N,dim_h)
 
            return c, h


    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _attention_layer_(self,name,shape_u,dim_u,dim_h,features,h,reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w1 = tf.get_variable('w1', [dim_h, dim_u], initializer=self.weight_initializer)
            w2 = tf.get_variable('w2', [dim_u, 1], initializer=self.weight_initializer)
            #w3 = tf.get_variable('w3', [dim_u, dim_u], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [dim_u], initializer=self.const_initializer)
            b2 = tf.get_variable('b2', [dim_u], initializer=self.const_initializer)

            # computer posterior higher attention  
            sparse = tf.nn.sigmoid(tf.matmul(h, w1)+b1)
            result0 = features * tf.expand_dims(sparse,1)   
            #out1 = tf.nn.relu(tf.matmul(tf.reshape(result0,[-1,dim_u]),w3)+b2)                   
            out1 = tf.nn.tanh(tf.reshape(result0,[-1,dim_u]) + b2)
            out2 = tf.nn.sigmoid(tf.reshape(tf.matmul(out1,w2),[-1,shape_u**2]))                    
            alpha = tf.nn.softmax(out2)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1)   
            return context, alpha
    

    def _attention_layer(self,name,alpha_l,shape_l,shape_u,dim_u,dim_h,features,h,reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w1 = tf.get_variable('w1', [dim_h, dim_u], initializer=self.weight_initializer)
            w2 = tf.get_variable('w2', [dim_u, 1], initializer=self.weight_initializer)
            #w3 = tf.get_variable('w3', [dim_u, dim_u], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [dim_u], initializer=self.const_initializer)
            b2 = tf.get_variable('b2', [dim_u], initializer=self.const_initializer)

            # resize lower attention to higher attention (prior higher attention)
            N = tf.shape(alpha_l)[0] # batch size
            alpha_ = tf.reshape(tf.transpose(alpha_l,perm=[1,0]),[shape_l,shape_l,N])  
            alpha_resized_ = tf.image.resize_images(alpha_,[shape_u,shape_u])          
            alpha_resized = tf.transpose(tf.reshape(alpha_resized_,[-1,N]),[1,0])      
            alpha_truncate = tf.nn.relu(alpha_resized-(1.0/shape_u**2))

            # computer posterior higher attention  
            sparse = tf.nn.sigmoid(tf.matmul(h, w1)+b1)
            result0 = features * tf.expand_dims(sparse,1)   
            #out1 = tf.nn.relu(tf.matmul(tf.reshape(result0,[-1,dim_u]),w3)+b2)                  
            out1 = tf.nn.tanh(tf.reshape(result0,[-1,dim_u]) + b2)
            out2 = tf.nn.sigmoid(tf.reshape(tf.matmul(out1,w2),[-1,shape_u**2]))
            alpha = tf.nn.softmax(out2+alpha_truncate)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1)   
            return context, alpha

    """
    def _combine_contexts(self,name,dim_c1,dim_c2,context1,context2,output_dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            w1 = tf.get_variable('w1', [dim_c1, output_dim], initializer=self.weight_initializer)
            w2 = tf.get_variable('w2', [dim_c2, output_dim], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [output_dim], initializer=self.const_initializer)
            b2 = tf.get_variable('b2', [output_dim], initializer=self.const_initializer)
            
            c1 = tf.matmul(context1,w1)+b1
            c2 = tf.matmul(context2,w2)+b2
            combined_context = tf.nn.tanh(c1+c2)

            return combined_context
    
    
    def _combine_contexts(self,name,dim_c1,dim_c2,context1,context2,output_dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            w = tf.get_variable('w', [dim_c1+dim_c2, output_dim], initializer=self.weight_initializer)
            b = tf.get_variable('b', [output_dim], initializer=self.const_initializer)
            w1 = tf.get_variable('w1', [dim_c1, dim_c1], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [dim_c1], initializer=self.const_initializer)
            w2 = tf.get_variable('w2', [dim_c2, dim_c2], initializer=self.weight_initializer)
            b2 = tf.get_variable('b2', [dim_c2], initializer=self.const_initializer)
            
            c1 = tf.nn.relu(tf.matmul(context1,w1)+b1)
            c2 = tf.nn.relu(tf.matmul(context2,w2)+b2)
            concated = tf.concat(1, [c1,c2])
            combined_context = tf.nn.tanh(tf.matmul(concated,w) + b)
            concated = tf.nn.dropout(concated,self.dropout_keep_prob)
            return combined_context
        

    
    
    def _combine_contexts(self,name,dim_c1,dim_c2,context1,context2,output_dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            w1 = tf.get_variable('w1', [dim_c1+dim_c2, output_dim], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [output_dim], initializer=self.const_initializer)
            w2 = tf.get_variable('w2', [output_dim, output_dim], initializer=self.weight_initializer)
            b2 = tf.get_variable('b2', [output_dim], initializer=self.const_initializer)
            
            c = tf.concat(1, [context1, context2])
            combined_context_ = tf.nn.relu(tf.matmul(c,w1)+b1)
            #combined_context_ = tf.nn.dropout(combined_context_, self.dropout_keep_prob)
            combined_context = tf.nn.tanh(tf.matmul(combined_context_,w2)+b2)
            return combined_context
    
    
    """
    def _combine_contexts(self,name,dim_c1,dim_c2,context1,context2,output_dim,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            w1 = tf.get_variable('w1', [dim_c1, output_dim], initializer=self.weight_initializer)
            w2 = tf.get_variable('w2', [dim_c2, output_dim], initializer=self.weight_initializer)
            b1 = tf.get_variable('b1', [output_dim], initializer=self.const_initializer)
            b2 = tf.get_variable('b2', [output_dim], initializer=self.const_initializer)
            
            c1 = tf.matmul(context1,w1)+b1
            c2 = tf.matmul(context2,w2)+b2
            combined_context = tf.nn.tanh(c1+c2)

            return combined_context
    
    def _decode_lstm(self,name,dim_h,dim_f,x,h,combined_context,reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            w_h = tf.get_variable('w_h', [dim_h, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            h = tf.nn.dropout(h, self.dropout_keep_prob)
            h_logits = tf.matmul(h, w_h) + b_h

            w_ctx2out = tf.get_variable('w_ctx2out', [dim_f, self.M], initializer=self.weight_initializer)
            h_logits += tf.matmul(combined_context, w_ctx2out)

            h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            h_logits = tf.nn.dropout(h_logits, self.dropout_keep_prob)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits



    def build_model(self):
        features1 = self.features_layer1
        features2 = self.features_layer2
        captions = self.captions

        batch_size = tf.shape(features1)[0]

        # process time input
        captions_in = captions[:, :self.T]  # including <START> excluding <END>    
        captions_out = captions[:, 1:]  # excluding <START> including <END>
        mask = tf.to_float(tf.not_equal(captions_out, self._null)) # mask out shorter sentence

        # initialize c & h for LSTM (use first layer)
        c, h = self._get_initial_lstm('initial_lstm',self.D1,self.H,features1)

        # get embedded words
        x = self._word_embedding(inputs=captions_in) 

        # store loss and attention prob
        loss = 0.0
        alpha_list1,alpha_list2 = [],[]
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H,state_is_tuple=True)
        entropy_list1,entropy_list2 = [],[]
        
        # rnn 
        for t in range(self.T):
            # extract image attention region & prob (multi-layer)
            
            context1, alpha1 = self._attention_layer_('att_layer1',self.S1,
                                                      self.D1,self.H,features1,h,reuse=(t!=0))
            context2, alpha2 = self._attention_layer('att_layer2',alpha1,self.S1,self.S2,
                                                      self.D2,self.H,features2,h,reuse=(t!=0))

            alpha_list1.append(alpha1) # T numbers of shape (N,L)
            alpha_list2.append(alpha2)
           
            # configure lstm
            with tf.variable_scope('lstm', reuse=(t!=0)):
                combined_context = self._combine_contexts('combined',self.D1,self.D2,context1,context2,512,reuse=(t!=0))
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x[:,t,:], combined_context]), state=[c, h])

            # decode output to sentence and compute loss
            logits = self._decode_lstm('logits',self.H,512,x[:,t,:],h,combined_context,reuse=(t!=0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])
         
            # entropy regularization
            entropy1 = tf.reduce_sum(-alpha1*tf.log(alpha1+self.e))
            entropy2 = tf.reduce_sum(-alpha2*tf.log(alpha2+self.e))
            entropy_list1.append(entropy1)
            entropy_list2.append(entropy2)

        # spreading regularization (focus on different region throughout time)
        alphas1 = tf.transpose(tf.pack(alpha_list1), (1, 0, 2))     # (N, T, L)
        alphas_all1 = tf.reduce_sum(alphas1, 1)      # (N, L)
        alphas2 = tf.transpose(tf.pack(alpha_list2), (1, 0, 2))     # (N, T, L)
        alphas_all2 = tf.reduce_sum(alphas2, 1)      # (N, L)
        entropys1 = tf.reduce_sum(tf.pack(entropy_list1)) 
        entropys2 = tf.reduce_sum(tf.pack(entropy_list2)) 
       
        loss += self.alpha_c[0]*tf.reduce_sum((1.0 - alphas_all1)**2)+self.alpha_c[1]*tf.reduce_sum((1.0 - alphas_all2)**2)     
        loss += self.alpha_e[0]*(entropys1/tf.to_float(self.T))+self.alpha_e[1]*(entropys2/tf.to_float(self.T)) 
       
        return (loss/tf.to_float(batch_size)),(entropys1/tf.to_float(self.T)),(entropys2/tf.to_float(self.T))




    def build_sampler(self, max_len=20):
        features1 = self.features_layer1
        features2 = self.features_layer2
        
        # init inputs for lstm
        c, h = self._get_initial_lstm('initial_lstm',self.D1,self.H,features1)
 
        sampled_word_list = []
        alpha_list1,alpha_list2 = [],[]

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H,state_is_tuple=True)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features1)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)  
            
            # extract image attention region & prob (multi-layer)
            #alpha_init = tf.fill([tf.shape(features1)[0],self.S1**2],1.0) 
            context1, alpha1 = self._attention_layer_('att_layer1',self.S1,
                                                      self.D1,self.H,features1,h,reuse=(t!=0))
            context2, alpha2 = self._attention_layer('att_layer2',alpha1,self.S1,self.S2,
                                                      self.D2,self.H,features2,h,reuse=(t!=0))
            alpha_list1.append(alpha1) # T numbers of shape (N,L)
            alpha_list2.append(alpha2)
  
            # configure lstm
            with tf.variable_scope('lstm', reuse=(t!=0)):
                combined_context = self._combine_contexts('combined',self.D1,self.D2,context1,context2,512,reuse=(t!=0))
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x,combined_context]), state=[c, h])
                
            logits = self._decode_lstm('logits',self.H,512,x,h,combined_context,reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)       
            sampled_word_list.append(sampled_word)     

        alphas1 = tf.transpose(tf.pack(alpha_list1), (1, 0, 2))     # (N, T, L)
        alphas2 = tf.transpose(tf.pack(alpha_list2), (1, 0, 2))     # (N, T, L)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))     # (N, max_len)

        return alphas1,alphas2,sampled_captions












