import tensorflow as tf

from exprgan import ExprGAN

flags = tf.app.flags
flags.DEFINE_integer(flag_name='epoch', default_value=50, docstring='number of epochs')
flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='number of batch size')
flags.DEFINE_integer(flag_name='y_dim', default_value=6, docstring='label dimension')
flags.DEFINE_integer(flag_name='rb_dim', default_value=5, docstring='expression intensity range')
flags.DEFINE_boolean(flag_name='is_train', default_value=True, docstring='training mode')
flags.DEFINE_string(flag_name='dataset', default_value='OULU', docstring='dataset name')
flags.DEFINE_string(flag_name='save_dir', default_value='save', docstring='dir for saving training results')
flags.DEFINE_string(flag_name='test_dir', default_value='test', docstring='dir for testing images')
flags.DEFINE_string(flag_name='checkpoint_dir', default_value='None', docstring='dir for loading checkpoints')
flags.DEFINE_boolean(flag_name='is_stage_one', default_value=True, docstring='is it the first stage?')
flags.DEFINE_float('vgg_coeff', 1.0, 'vgg coefficient')
flags.DEFINE_float('q_coeff', 1.0, 'regularizer network coefficient')
flags.DEFINE_float('fm_coeff', 0.0, 'feature matching coefficient')
FLAGS = flags.FLAGS


def main(_):
    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = ExprGAN(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            is_stage_one=FLAGS.is_stage_one,
            size_batch=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            rb_dim=FLAGS.rb_dim,
            vgg_coeff=FLAGS.vgg_coeff,
            q_coeff=FLAGS.q_coeff,
            fm_coeff=FLAGS.fm_coeff
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
            )
        else:
            seed = 2018
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=None, random_seed=seed
            )


if __name__ == '__main__':

    tf.app.run()

