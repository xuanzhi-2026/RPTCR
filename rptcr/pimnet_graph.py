"""TensorFlow 1 graph adapter for PIMNet verification.

The caller supplies TensorFlow and a factory for the upstream PIMNet model.
"""
from .pimnet import T


class Capture(object):
    def __init__(self, original):
        self.original = original

    def __call__(self, images):
        outputs = self.original(images)
        self.feature = outputs[1]
        return outputs


def build_graph(tf, model_factory, seed=20260905):
    """Build five-step native decoding and verification with shared weights.

    model_factory(num_iter) must construct the upstream Model with its official
    LOWERCASE vocabulary/configuration, seq_len=25, and is_training=False.
    It is called inside the new graph. `resue` is the upstream API spelling.
    """
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)
        images = tf.placeholder(tf.float32, [1, 64, 256, 1], name='input_images')
        labels = tf.placeholder(tf.int32, [1, 25], name='input_labels')
        model = model_factory(T)
        model.backbone = Capture(model.backbone)
        native_logits, _, native, _, _, _, _, _ = model(images, labels, reuse=False)
        before = [(v.name, v.shape.as_list()) for v in tf.trainable_variables()]
        features = model.backbone.feature
        supplied = tf.placeholder(tf.float32, features.shape.as_list(), name='cached_feature')
        tokens = tf.placeholder(tf.int32, [1, 25], name='terminal_tokens')
        with tf.variable_scope('model', reuse=True):
            positions = model.position_embedding(resue=True)
            with tf.variable_scope('iterative_decoder', reuse=True):
                read_logits = model.iterative_decoder.decoder(
                    tokens, supplied, pos_embedding=positions, reuse=True)[0]
        after = [(v.name, v.shape.as_list()) for v in tf.trainable_variables()]
        if before != after:
            raise RuntimeError('Terminal reread introduced trainable variables')
        step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                               trainable=False, dtype=tf.int32)
        ema = tf.train.ExponentialMovingAverage(0.997, step)
        saver = tf.train.Saver(ema.variables_to_restore())
    return {'graph': graph, 'images': images, 'labels': labels, 'native': native,
            'native_logits': native_logits, 'features': features, 'supplied': supplied,
            'tokens': tokens, 'read_logits': read_logits, 'trainable_count': len(before),
            'saver': saver}


def open_session(tf, graph, checkpoint):
    """Restore the EMA checkpoint in a CPU session."""
    config = tf.ConfigProto(allow_soft_placement=False, device_count={'GPU': 0},
                            intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(graph=graph['graph'], config=config)
    try:
        graph['saver'].restore(sess, str(checkpoint))
    except Exception:
        sess.close()
        raise
    return sess
