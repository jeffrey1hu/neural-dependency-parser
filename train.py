import os
import time
import cPickle
import tensorflow as tf

from config import Config
from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data


def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = ParserModel(config, embeddings)
        parser.model = model
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            parser.session = session
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            model.fit(session, saver, parser, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/parser.weights')
                print "Final evaluation on test set",
                UAS, dependencies = parser.parse(test_set)
                print "- test UAS: {:.2f}".format(UAS * 100.0)
                print "Writing predictions"
                with open('q2_test.predicted.pkl', 'w') as f:
                    cPickle.dump(dependencies, f, -1)
                print "Done!"

if __name__ == '__main__':
    main(False)
