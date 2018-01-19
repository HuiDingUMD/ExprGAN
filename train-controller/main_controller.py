import tensorflow as tf
from controller import Controller

flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=200, docstring='number of epochs')
flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='number of epochs')
flags.DEFINE_integer(flag_name='y_dim', default_value=8, docstring='number of epochs')
flags.DEFINE_integer(flag_name='rb_dim', default_value=5, docstring='number of epochs')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='dataset', default_value='RAF', docstring='dataset name')
flags.DEFINE_string(flag_name='save_dir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_string(flag_name='test_dir', default_value='test', docstring='dir for testing images')
flags.DEFINE_string(flag_name='checkpoint_dir', default_value='None', docstring='dir for loading checkpoints')
flags.DEFINE_boolean(flag_name='is_vis', default_value=False, docstring='is it the first stage?')
flags.DEFINE_boolean(flag_name='is_simple_q', default_value=False, docstring='is it the first stage?')
flags.DEFINE_integer(flag_name='z_dim', default_value=50, docstring='number of epochs')
FLAGS = flags.FLAGS


def main(_):

    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = Controller(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            size_batch=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            rb_dim=FLAGS.rb_dim,
            is_simple_q=FLAGS.is_simple_q,
            num_z_channels=FLAGS.z_dim
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,
            )
        else:
            print '\n\tVisualization Mode'
            model.custom_visualize(testing_samples_dir=FLAGS.test_dir + '/*.jpeg')



if __name__ == '__main__':

    tf.app.run()


