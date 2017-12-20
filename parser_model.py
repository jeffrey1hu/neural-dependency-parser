import tensorflow as tf

from model import Model
from utils.general_utils import Progbar
from utils.parser_utils import minibatches

class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_features), name='input_placeholder')
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_classes), name='labels_placeholder')
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None), name='droupout')
        self.learning_rate = tf.placeholder(tf.float32, name='lr')
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        feed_dict = dict()
        feed_dict[self.dropout_placeholder] = self.config.dropout
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.learning_rate] = self.config.lr
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        # shape -> batch_size * n_feature * embed_size
        emb = tf.Variable(self.pretrained_embeddings)
        _input_embeddings = tf.nn.embedding_lookup(emb, self.input_placeholder)
        assert (_input_embeddings.get_shape()[1], _input_embeddings.get_shape()[2]) == (self.config.n_features, self.config.embed_size)
        embeddings = tf.reshape(_input_embeddings, shape=(-1, self.config.n_features * self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument. 
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        regularizer = tf.contrib.layers.l2_regularizer(self.config.reg)
        ### YOUR CODE HERE
        with tf.variable_scope("nn"):
            W = tf.get_variable("W",
                                shape=(self.config.n_features * self.config.embed_size, self.config.hidden_size),
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=regularizer)
            b1 = tf.get_variable("b1",
                                 shape=(self.config.hidden_size, ),
                                 initializer=tf.zeros_initializer(),
                                 regularizer=regularizer,
                                 dtype=tf.float32)
            U = tf.get_variable("U",
                                shape=(self.config.hidden_size, self.config.n_classes),
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=regularizer)
            b2 = tf.get_variable('b2',
                                 shape=(self.config.n_classes, ),
                                 initializer=tf.zeros_initializer(),
                                 regularizer=regularizer,
                                 dtype=tf.float32)
            h = tf.add(tf.matmul(x, W), b1)
            cube_h = h ** 3
            h_drop = tf.nn.dropout(cube_h, keep_prob=self.dropout_placeholder)
            pred = tf.add(tf.matmul(h_drop, U), b2)
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularizer = tf.contrib.layers.l2_regularizer(self.config.reg)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        ### END YOUR CODE
        return loss + reg_term

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.config.lr_decay[0], self.config.lr_decay[1],
                                                   staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        ### END YOUR CODE
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, parser, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print "New best dev UAS! Saving model in ./data/weights/parser.weights"
                    saver.save(sess, './data/weights/parser.weights')
            print

    def __init__(self, config, pretrained_embeddings):
        self.global_step = tf.Variable(0, trainable=False)
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


