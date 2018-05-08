import tensorflow as tf
import os

class RnnModel:
    def __init__(self, is_training, image_dim, vocab_size, N_hidden, N_video_step, N_caption_step, **params):

        self.is_training = is_training
        self.image_dim = image_dim # feature_dim: 4096
        self.vocab_size = vocab_size # vocab size: len(dictionary)
        self.N_hidden = N_hidden # hidden layer size (hidden state dim)
        self.N_video_step = N_video_step # time_step: 80
        self.N_caption_step = N_caption_step # max sequence length

        self.cell_type = params['cell_type']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.hidden_layers = params['hidden_layers']
        self.dropout = params['dropout']


        if self.cell_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.cell_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif self.cell_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(cell_fn(self.N_hidden, reuse= tf.get_variable_scope().reuse))

        if is_training and self.dropout > 0:
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob= 1.0 - self.dropout) for cell in cells]
        self.encoder_multi_cells = tf.contrib.rnn.MultiRNNCell(cells)

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(cell_fn(self.N_hidden, reuse= tf.get_variable_scope().reuse))

        if is_training and self.dropout > 0:
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob= 1.0 - self.dropout) for cell in cells]
        self.decoder_multi_cells = tf.contrib.rnn.MultiRNNCell(cells)


        self.image_weight = tf.get_variable('image_weight',
                                  shape= (self.image_dim, self.N_hidden),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.image_bias = tf.get_variable('image_bias',
                                  shape= (self.N_hidden),
                                  initializer= tf.constant_initializer())
        self.word_emdeded = tf.get_variable('word_emdeded',
                                  shape= (self.vocab_size, self.N_hidden),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.word_weight = tf.get_variable('word_weight',
                                  shape= (self.N_hidden, self.vocab_size),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.word_bias = tf.get_variable('word_bias',
                                  shape= (self.vocab_size),
                                  initializer= tf.constant_initializer())
        self.saver = None



    def build_train_model(self):
        # Inputs
        video = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_video_step, self.image_dim])

        decoder_input = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])
        decoder_target = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])

        # Embeded image_feat size to N_hidden
        video_flatten = tf.reshape(video, (-1, self.image_dim))
        image_embeded = tf.matmul(video_flatten, self.image_weight) + self.image_bias
        image_embeded = tf.reshape(image_embeded, (-1, self.N_video_step, self.N_hidden))

        # RNN parameters
        state = self.encoder_multi_cells.zero_state(self.batch_size, dtype= tf.float32)

        # Encoding Stage
        with tf.variable_scope('encoder', reuse= tf.get_variable_scope().reuse):
            encoder_outputs, state = tf.nn.dynamic_rnn(self.encoder_multi_cells, image_embeded, initial_state= state)

        # Decoding Stage
        decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, decoder_input)
        with tf.variable_scope('decoder', reuse= tf.get_variable_scope().reuse):     
            decoder_output, _ = tf.nn.dynamic_rnn(self.decoder_multi_cells, decoder_input_embeded, initial_state= state)
                        
        # Project N_hidden into vocab_size
        decoder_output_flatten = tf.reshape(decoder_output, (-1, self.N_hidden))
        decoder_logits = tf.matmul(decoder_output_flatten, self.word_weight) + self.word_bias
        decoder_logits = tf.reshape(decoder_logits, (self.batch_size, -1, self.vocab_size))

        # Loss
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels= tf.one_hot(decoder_target, depth= self.vocab_size, dtype=tf.float32),
                logits= decoder_logits)
        
        loss = tf.reduce_mean(tf.reduce_sum(stepwise_cross_entropy, axis= 1))
        optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_step = optimizer.apply_gradients(clipped_gradients)

        return video, decoder_input, decoder_target, loss, train_step
 
    
    def build_test_model(self, sampling= False):
        # Inputs
        video = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_video_step, self.image_dim])

        decoder_input = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])

        # Embeded image_feat size to N_hidden
        video_flatten = tf.reshape(video, (-1, self.image_dim))
        image_embeded = tf.matmul(video_flatten, self.image_weight) + self.image_bias
        image_embeded = tf.reshape(image_embeded, (-1, self.N_video_step, self.N_hidden))

        # RNN parameters
        state = self.encoder_multi_cells.zero_state(self.batch_size, dtype= tf.float32)
        captions = []
        
        # Encoding Stage
        with tf.variable_scope('encoder', reuse= tf.get_variable_scope().reuse):
            encoder_outputs, state = tf.nn.dynamic_rnn(self.encoder_multi_cells, image_embeded, initial_state= state)

        # Decoding Stage
        decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, decoder_input)
        for idx in range(self.N_caption_step):
            with tf.variable_scope('decoder', reuse= tf.get_variable_scope().reuse):     
                decoder_output, state = tf.nn.dynamic_rnn(self.decoder_multi_cells, decoder_input_embeded, initial_state= state)
                            
            # Project N_hidden into vocab_size
            decoder_output_flatten = tf.reshape(decoder_output, (-1, self.N_hidden))
            decoder_logits = tf.matmul(decoder_output_flatten, self.word_weight) + self.word_bias
            decoder_logits = tf.reshape(decoder_logits, (self.batch_size, self.vocab_size))
            
            probs = tf.nn.softmax(decoder_logits)
            if sampling:
                best_choice = tf.multinomial(tf.log(probs), 1)
            else:
                best_choice = tf.expand_dims(tf.argmax(decoder_logits, axis= 1), axis= -1)
            best_choice = tf.cast(best_choice, dtype= tf.int32)
            
            decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, best_choice)
            captions.append(best_choice)
            
        captions = tf.squeeze(tf.stack(captions, axis= 1))
        return video, decoder_input, captions
    

    def save_model(self, sess, model_file, step):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep= 5)
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        self.saver.save(sess, model_file, global_step= step)

    def restore_model(self, sess, model_file):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep= 5)
        step = 0
        checkpoint_dir = os.path.dirname(model_file)
        if os.path.isdir(checkpoint_dir):
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            step = int( checkpoint.model_checkpoint_path.split("-")[1].split(".")[0])
            self.saver.restore(sess,checkpoint.model_checkpoint_path)
        return step

#%%

class RnnModel_Attention:
    def __init__(self, is_training, image_dim, vocab_size, N_hidden, N_video_step, N_caption_step, **params):

        self.is_training = is_training
        self.image_dim = image_dim # feature_dim: 4096
        self.vocab_size = vocab_size # vocab size: len(dictionary)
        self.N_hidden = N_hidden # hidden layer size (hidden state dim)
        self.N_video_step = N_video_step # time_step: 80
        self.N_caption_step = N_caption_step # max sequence length

        self.cell_type = params['cell_type']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.hidden_layers = params['hidden_layers']
        self.dropout = params['dropout']


        if self.cell_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.cell_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif self.cell_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(cell_fn(self.N_hidden, reuse= tf.get_variable_scope().reuse))

        if is_training and self.dropout > 0:
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob= 1.0 - self.dropout) for cell in cells]
        self.encoder_multi_cells = tf.contrib.rnn.MultiRNNCell(cells)

        cells = []
        for _ in range(self.hidden_layers):
            cells.append(cell_fn(self.N_hidden, reuse= tf.get_variable_scope().reuse))

        if is_training and self.dropout > 0:
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob= 1.0 - self.dropout) for cell in cells]
        self.decoder_multi_cells = tf.contrib.rnn.MultiRNNCell(cells)


        self.image_weight = tf.get_variable('image_weight',
                                  shape= (self.image_dim, self.N_hidden),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.image_bias = tf.get_variable('image_bias',
                                  shape= (self.N_hidden),
                                  initializer= tf.constant_initializer())
        self.word_emdeded = tf.get_variable('word_emdeded',
                                  shape= (self.vocab_size, self.N_hidden),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.word_weight = tf.get_variable('word_weight',
                                  shape= (self.N_hidden, self.vocab_size),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))
        self.word_bias = tf.get_variable('word_bias',
                                  shape= (self.vocab_size),
                                  initializer= tf.constant_initializer())
        self.saver = None
        self.key = tf.get_variable('key',
                                  shape= (self.vocab_size),
                                  initializer= tf.truncated_normal_initializer(stddev= 0.02))

    def attention(self, states, key):
        states_flatten = tf.reshape(states_flatten, (-1, self.N_hidden))
        states_flatten_normalized = tf.nn.l2_normalize(states_flatten, 1)
        states_flatten_normalized = tf.matmul(states_flatten_normalized, tf.nn.l2_normalize(self.key))
        
        states_flatten_normalized = tf.reshape(
            states_flatten_normalized, 
            (self.batch_size, -1, self.N_hidden)
        )
        
        
        


    def build_train_model(self):
        # Inputs
        video = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_video_step, self.image_dim])

        decoder_input = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])
        decoder_target = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])

        # Embeded image_feat size to N_hidden
        video_flatten = tf.reshape(video, (-1, self.image_dim))
        image_embeded = tf.matmul(video_flatten, self.image_weight) + self.image_bias
        image_embeded = tf.reshape(image_embeded, (-1, self.N_video_step, self.N_hidden))

        # RNN parameters
        state = self.encoder_multi_cells.zero_state(self.batch_size, dtype= tf.float32)

        # Encoding Stage
        with tf.variable_scope('encoder', reuse= tf.get_variable_scope().reuse):
            encoder_outputs, state = tf.nn.dynamic_rnn(self.encoder_multi_cells, image_embeded, initial_state= state)

        # Decoding Stage
        decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, decoder_input)
        with tf.variable_scope('decoder', reuse= tf.get_variable_scope().reuse):     
            decoder_output, _ = tf.nn.dynamic_rnn(self.decoder_multi_cells, decoder_input_embeded, initial_state= state)
                        
        # Project N_hidden into vocab_size
        decoder_output_flatten = tf.reshape(decoder_output, (-1, self.N_hidden))
        decoder_logits = tf.matmul(decoder_output_flatten, self.word_weight) + self.word_bias
        decoder_logits = tf.reshape(decoder_logits, (self.batch_size, -1, self.vocab_size))

        # Loss
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels= tf.one_hot(decoder_target, depth= self.vocab_size, dtype=tf.float32),
                logits= decoder_logits)
        
        loss = tf.reduce_mean(tf.reduce_sum(stepwise_cross_entropy, axis= 1))
        optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_step = optimizer.apply_gradients(clipped_gradients)

        return video, decoder_input, decoder_target, loss, train_step
 
    
    def build_test_model(self, sampling= False):
        # Inputs
        video = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_video_step, self.image_dim])

        decoder_input = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, None])

        # Embeded image_feat size to N_hidden
        video_flatten = tf.reshape(video, (-1, self.image_dim))
        image_embeded = tf.matmul(video_flatten, self.image_weight) + self.image_bias
        image_embeded = tf.reshape(image_embeded, (-1, self.N_video_step, self.N_hidden))

        # RNN parameters
        state = self.encoder_multi_cells.zero_state(self.batch_size, dtype= tf.float32)
        captions = []
        
        # Encoding Stage
        with tf.variable_scope('encoder', reuse= tf.get_variable_scope().reuse):
            encoder_outputs, state = tf.nn.dynamic_rnn(self.encoder_multi_cells, image_embeded, initial_state= state)

        # Decoding Stage
        decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, decoder_input)
        for idx in range(self.N_caption_step):
            with tf.variable_scope('decoder', reuse= tf.get_variable_scope().reuse):     
                decoder_output, state = tf.nn.dynamic_rnn(self.decoder_multi_cells, decoder_input_embeded, initial_state= state)
                            
            # Project N_hidden into vocab_size
            decoder_output_flatten = tf.reshape(decoder_output, (-1, self.N_hidden))
            decoder_logits = tf.matmul(decoder_output_flatten, self.word_weight) + self.word_bias
            decoder_logits = tf.reshape(decoder_logits, (self.batch_size, self.vocab_size))
            
            probs = tf.nn.softmax(decoder_logits)
            if sampling:
                best_choice = tf.multinomial(tf.log(probs), 1)
            else:
                best_choice = tf.expand_dims(tf.argmax(decoder_logits, axis= 1), axis= -1)
            best_choice = tf.cast(best_choice, dtype= tf.int32)
            
            decoder_input_embeded = tf.nn.embedding_lookup(self.word_emdeded, best_choice)
            captions.append(best_choice)
            
        captions = tf.squeeze(tf.stack(captions, axis= 1))
        return video, decoder_input, captions
    

    def save_model(self, sess, model_file, step):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep= 5)
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        self.saver.save(sess, model_file, global_step= step)

    def restore_model(self, sess, model_file):
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep= 5)
        step = 0
        checkpoint_dir = os.path.dirname(model_file)
        if os.path.isdir(checkpoint_dir):
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            step = int( checkpoint.model_checkpoint_path.split("-")[1].split(".")[0])
            self.saver.restore(sess,checkpoint.model_checkpoint_path)
        return step