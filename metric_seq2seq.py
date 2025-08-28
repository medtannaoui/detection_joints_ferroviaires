import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class PrecisionSeq(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='precision_per_class', mode='frame', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mode = mode
        self.precisions = [Precision(name=f'precision_class_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.mode == 'frame':
            y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            y_true_labels = tf.argmax(y_true_flat, axis=1)
            y_pred_labels = tf.argmax(y_pred_flat, axis=1)
        else:
            y_true_labels = tf.argmax(y_true, axis=-1)
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            y_true_labels = tf.cast(tf.reduce_all(tf.equal(y_true_labels, y_pred_labels), axis=1), tf.int32)
            y_pred_labels = tf.ones_like(y_true_labels)

        for i in range(self.num_classes):
            y_true_i = tf.cast(tf.equal(y_true_labels, i), tf.int32)
            y_pred_i = tf.cast(tf.equal(y_pred_labels, i), tf.int32)
            self.precisions[i].update_state(y_true_i, y_pred_i, sample_weight)

    def result(self):
        return {f'precision_class_{i}': self.precisions[i].result() for i in range(1,self.num_classes)}

    def reset_states(self):
        for metric in self.precisions:
            metric.reset_states()

@register_keras_serializable()
class RecallSeq(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='recall_per_class', mode='frame', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mode = mode
        self.recalls = [Recall(name=f'recall_class_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.mode == 'frame':
            y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            y_true_labels = tf.argmax(y_true_flat, axis=1)
            y_pred_labels = tf.argmax(y_pred_flat, axis=1)
        else:
            y_true_labels = tf.argmax(y_true, axis=-1)
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            y_true_labels = tf.cast(tf.reduce_all(tf.equal(y_true_labels, y_pred_labels), axis=1), tf.int32)
            y_pred_labels = tf.ones_like(y_true_labels)

        for i in range(1,self.num_classes):
            y_true_i = tf.cast(tf.equal(y_true_labels, i), tf.int32)
            y_pred_i = tf.cast(tf.equal(y_pred_labels, i), tf.int32)
            self.recalls[i].update_state(y_true_i, y_pred_i, sample_weight)

    def result(self):
        return {f'recall_class_{i}': self.recalls[i].result() for i in range(1,self.num_classes)}

    def reset_states(self):
        for metric in self.recalls:
            metric.reset_states()

@register_keras_serializable()
class F1ScoreSeq(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_per_class', mode='frame', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mode = mode
        self.precisions = [Precision() for _ in range(num_classes)]
        self.recalls = [Recall() for _ in range(num_classes)]
        self.supports = self.add_weight(shape=(num_classes,), initializer="zeros", name="supports")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.mode == 'frame':
            y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
            y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
            y_true_labels = tf.argmax(y_true_flat, axis=1)
            y_pred_labels = tf.argmax(y_pred_flat, axis=1)
        else:
            y_true_labels = tf.argmax(y_true, axis=-1)
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            y_true_labels = tf.cast(tf.reduce_all(tf.equal(y_true_labels, y_pred_labels), axis=1), tf.int32)
            y_pred_labels = tf.ones_like(y_true_labels)

        for i in range(1,self.num_classes):
            y_true_i = tf.cast(tf.equal(y_true_labels, i), tf.int32)
            y_pred_i = tf.cast(tf.equal(y_pred_labels, i), tf.int32)
            self.precisions[i].update_state(y_true_i, y_pred_i, sample_weight)
            self.recalls[i].update_state(y_true_i, y_pred_i, sample_weight)
            count_i = tf.reduce_sum(tf.cast(tf.equal(y_true_labels, i), tf.float32))
            indices = tf.constant([[i]])
            self.supports.assign(tf.tensor_scatter_nd_add(self.supports, indices, tf.reshape(count_i, [1])))

    def result(self):
        results = {}
        f1_scores = []
        #total_support = tf.reduce_sum(self.supports) + 1e-8

        for i in range(1,self.num_classes):
            p = self.precisions[i].result()
            r = self.recalls[i].result()
            f1 = tf.cond(tf.equal(p + r, 0), lambda: 0.0, lambda: 2 * (p * r) / (p + r))
            results[f'f1_class_{i}'] = f1
            f1_scores.append(f1)

        

        return results

    def reset_states(self):
        for i in range(self.num_classes):
            self.precisions[i].reset_states()
            self.recalls[i].reset_states()
        self.supports.assign(tf.zeros_like(self.supports))


@register_keras_serializable()
class AUCSeq(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='auc_per_class', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        if num_classes == 2:
            self.auc = AUC(name="auc_binary")
        else:
            self.aucs = [AUC(name=f'auc_class_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_flat = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])

        if self.num_classes == 2:
            self.auc.update_state(y_true_flat[:, 1], y_pred_flat[:, 1], sample_weight)
        else:
            for i in range(self.num_classes):
                self.aucs[i].update_state(y_true_flat[:, i], y_pred_flat[:, i], sample_weight)

    def result(self):
        if self.num_classes == 2:
            return {'auc_binary': self.auc.result()}
        else:
            aucs = [self.aucs[i].result() for i in range(self.num_classes)]
            results = {f'auc_class_{i}': aucs[i] for i in range(self.num_classes)}
            results['auc_macro'] = tf.reduce_mean(aucs)
            return results

    def reset_states(self):
        if self.num_classes == 2:
            self.auc.reset_states()
        else:
            for auc in self.aucs:
                auc.reset_states()
