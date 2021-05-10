import tensorflow as tf


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

# 0.3132617 -> 0.6931471
res = tf.reduce_mean(
    sigmoid_cross_entropy_with_logits(tf.ones((500, 1)), tf.ones((500, 1))))

with tf.Session() as sess:
    print(res.eval())
