from inits import *
# import tensorflow as tf
import tensorflow._api.v1.compat.v1 as tf
import numpy as np
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.test = []
        self.matepath_construction = None

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphContrastLayer(Layer):

    def __init__(self,Contrast_depth,placeholders, tag,**kwargs):
        super(GraphContrastLayer,self).__init__(**kwargs)
        self.Contrast_depth = Contrast_depth
        self.tag = tag
        self.placeholders = placeholders
        self.support = placeholders['support_' + tag]
        self.struc_los = None

    def get_los(self, initial_emb,struc_emb,matepath_num):
        totol_los = 0
        # print("initial_emb形状", initial_emb.shape)
        # print("initial_embtype", type(initial_emb))
        # print("struc_embtype", type(struc_emb))
        # print("struc_emb形状", struc_emb.shape)
        for i in range(matepath_num):

            # print("type:",type(initial_emb))
            initial_pos = tf.nn.embedding_lookup(initial_emb[i], self.placeholders["pos_" + self.tag])
            # initial_neg = tf.nn.embedding_lookup(initial_emb[i], self.placeholders["neg_" + self.tag])
            struc_pos = tf.nn.embedding_lookup(struc_emb[i], self.placeholders["pos_" + self.tag])

            initial_neg = tf.nn.embedding_lookup(initial_emb[i], self.placeholders["neg_" + self.tag][i])

            # struc_neg = tf.nn.embedding_lookup(struc_emb[i], self.placeholders["neg_" + self.tag][i])

            L2_initial_pos = tf.nn.l2_normalize(initial_pos,axis=2)
            # L2_initial_neg = tf.nn.l2_normalize(initial_neg,axis=2)
            L2_struc_pos = tf.nn.l2_normalize(struc_pos,axis=2)
            L2_initial_neg = tf.nn.l2_normalize(initial_neg,axis=2)
            pos_cos = get_cosine_similarity(L2_struc_pos,L2_initial_pos)
            neg_cos = get_cosine_similarity(L2_struc_pos, L2_initial_neg)
            totol_los += (-tf.reduce_mean(tf.log(new_softmax(pos_cos, neg_cos, FLAGS.tau))))

        return totol_los


    def _call(self,imputs):
        # imputs:图卷积的输出，多条元路径的embedding shape（n，itemsize，64）
        # 要取成 正（1024，n，1，64） 负（1024，n，neg，64）计算loss
        # query_emb = tf.nn.embedding_lookup(imputs,self.placeholders["query_"+self.tag])
        # pos_emb = tf.nn.embedding_lookup(imputs,self.placeholders["pos_"+self.tag])
        # neg_emb = tf.nn.embedding_lookup(imputs, self.placeholders["neg_" + self.tag])
        #计算出经过n层后的embedding
        output = [None for i in range(self.Contrast_depth+1)]
        output[0]=tf.nn.dropout(imputs,1-self.placeholders["dropout"])
        print("add struc dropout")
        # print("形状",imputs.shape)
        # print("type", type(imputs))
        for i in range(self.Contrast_depth):
            temp = [None for i in range(len(self.support))]
            for j in range(len(self.support)):
                A = self.support[j]
                temp[j] = tf.matmul(A,output[i][j])
            output[i+1] = temp
        self.struc_los = self.get_los(imputs,output[self.Contrast_depth],len(self.support))
        return imputs





class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, length, placeholders, tag, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        """
        copilot
        params:
            input_dim: dimensionality of input features
            output_dim: dimensionality of output features
            length: length of the input sequence
            placeholders: placeholder dictionary
            tag: tag for the variable scope
            dropout: dropout probability
            sparse_inputs: if True, gradient updates will be performed on
        """
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support_'+tag]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.tag = tag
        self.length = length

        # helper variable for sparse dropout
        # self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_' + self.tag + '_vars'):
            for i in range(len(self.support)):
                if not self.featureless:
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
                else:
                    # print(placeholders["features_{}".format(self.tag)].shape[0])
                    self.vars['weights_' + str(i)] = glorot([int(placeholders["features_{}".format(self.tag)].shape[0]),
                                                             output_dim], name='weights_' + str(i))
                print("weights_{} dim:{},{}".format(i, self.vars['weights_' + str(i)].shape[0],
                                                    self.vars['weights_' + str(i)].shape[1]))
                # self.vars['bias_'+str(i)] = zeros([output_dim,], name='bias_' + str(i))
                self.vars['bias_' + str(i)] = tf.zeros(shape=(self.length, 1), name='bias_' + str(i))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        """
        inputs: H(l)
        outputs: H(l+1)
        """
        print("GCN _call inputs shape", inputs.shape)
        supports = list()
        for i in range(len(self.support)):
            print("Processing {}-th support_{}".format(i, self.tag))
            if self.name == 'first'+self.tag: #这里注释了
                print("Name including first{}, x=inputs".format(self.tag))
                x = inputs
            else:
                x = inputs[i]
            # x = inputs #做成concat需要修改三个地方，修改这里的输入，add输出，移除attention

            # dropout
            x = tf.nn.dropout(x, 1-self.dropout)

            print("x shape", x.shape)

        # convolve
        #     support = tf.matmul(self.support[i], x)
            if not self.featureless:
                """ Here if using content features """
                pre_sup = dot(x, self.vars['weights_' + str(i)])
            else:
                """ If not, use weights for training """
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup)
            # self.test.append(self.vars['bias_' + str(i)])
            support = support + self.vars['bias_' + str(i)]
            supports.append(self.act(support))
        # output = tf.add_n(supports) #这里解除注释了
        output = supports #这里注释了
        print("GCN output", len(output))
        # bias
        # return output
        return self.act(output) # support already had self.act(support) why again? #确实


class RatLayer():
    def __init__(self, user, item, act=tf.nn.relu):
        self.user = user
        self.item = item
        self.act = act

    def __call__(self):
        rate_matrix = tf.matmul(self.user, tf.transpose(self.item))
        return self.act(rate_matrix)





class PredictLayer():
    def __init__(self,placeholders,user_dim,item_dim,user,item,tua,act=tf.nn.relu):
        self.user = tf.nn.l2_normalize(user,1)
        self.item = tf.nn.l2_normalize(item,1)
        self.tua = tua
        self.train_samples = placeholders['train_samples']
        self.uids = placeholders['uids']
        self.name = 'PredictLayer'
        self.vars = {}
        self.act = act
        with tf.name_scope(self.name + '_vars'):
            self.vars["user_latent"] = init_variable(user_dim, int(FLAGS.latent_dim), name='user_latent_matrix')
            self.vars["item_latent"] = init_variable(item_dim, int(FLAGS.latent_dim), name='item_latent_matrix')
            self.vars['item_bias'] = init_variable(item_dim, 1, "item_bias")
            self.vars['user_alpha1'] = tf.Variable(initial_value=1., name='user_alpha1')
            self.vars['item_alpha1'] = tf.Variable(initial_value=1., name='item_alpha1')
            # self.vars['alpha2'] = tf.Variable(initial_value=1., name='alpha2')
            self.vars['user_weight'] = glorot([int(FLAGS.output_dim), int(FLAGS.latent_dim)], name='user_weight')
            self.vars['item_weight'] = glorot([int(FLAGS.output_dim), int(FLAGS.latent_dim)], name='item_weight')
            self.vars['user_bias'] = zeros(FLAGS.latent_dim,name='user_bias')
            self.vars['item_bias'] = zeros(FLAGS.latent_dim, name='item_bias')

    def __call__(self):
        # uids = self.uids
        # print("uids shape", uids.shape)
        # samples = self.train_samples[:, 1:]
        # print("samples shape", samples.shape)
        # # user_emb shape: [batch_size, user_dim]
        # user_emb = tf.nn.embedding_lookup(self.user, uids)
        # # item_emb shape: [batch_size,user_dim, item_dim]
        # item_emb = tf.nn.embedding_lookup(self.item, samples)
        # print("user_emb shape", user_emb.shape)
        # print("item_emb shape", item_emb.shape)
        #
        # batch_cos = get_cosine_similarity(user_emb, item_emb)
        # # x = tf.nn.l2_normalize(x)
        # # y = tf.nn.l2_normalize(y, dim=1)
        # # return tf.reduce_sum(tf.multiply(x, y), 1)
        # # user_emb shape: [ user_dim,user_emb_dim]
        # # item_emb shape: [ item_dim,item_emb_dim]
        # usr = tf.nn.l2_normalize(self.user, axis=1)
        # item = tf.nn.l2_normalize(self.item, axis=1)
        # rating = tf.matmul(usr, tf.transpose(item))
        uids = self.uids
        samples = self.train_samples[:, 1:]
        user_emb_1 = tf.expand_dims(tf.nn.embedding_lookup(self.user, uids),3)
        item_emb_1 = tf.expand_dims(tf.nn.embedding_lookup(self.item, samples),3)
        W_user = tf.expand_dims(tf.expand_dims(self.vars['user_weight'],0),0)
        W_item = tf.expand_dims(tf.expand_dims(self.vars['item_weight'],0),0)
        # print("W_user shape", W_user.shape)
        # print("W_item shape", W_item.shape)
        #
        #
        #
        # print("user_emb_1 shape", user_emb_1.shape)
        # print("item_emb_1 shape", item_emb_1.shape)
        # print("uids shape", uids.shape)
        user_emb_1 = self.act(tf.reduce_sum(W_user * user_emb_1,axis = 2) + self.vars['user_bias'])
        item_emb_1 = self.act(tf.reduce_sum(W_item * item_emb_1,axis = 2) + self.vars['item_bias'])
        print("user_emb_1 shape", user_emb_1.shape)
        print("item_emb_1 shape", item_emb_1.shape)
        # user_emb_1 = tf.nn.relu(tf.matmul(user_emb_1, self.vars['user_weight']) + self.vars['user_bias'])
        # item_emb_1 = tf.nn.relu(tf.matmul(item_emb_1, self.vars['item_weight']) + self.vars['item_bias'])

        user_emb_2 = tf.nn.embedding_lookup(self.vars["user_latent"], uids)
        item_emb_2 = tf.nn.embedding_lookup(self.vars["item_latent"], samples)
        # bias = tf.nn.embedding_lookup(self.vars['item_bias'], samples)
        user_emb = self.vars['user_alpha1']*user_emb_1 + user_emb_2
        item_emb = self.vars['item_alpha1']*item_emb_1 + item_emb_2
        batch_cos = get_cosine_similarity(user_emb, item_emb)
        # batch_cos = 1



        user_1 = self.act(tf.matmul(self.user, self.vars['user_weight'])+self.vars['user_bias'])
        item_1 = self.act(tf.matmul(self.item, self.vars['item_weight'])+self.vars['item_bias'])
        user = self.vars['user_alpha1']*user_1 + self.vars['user_latent']
        item = self.vars['item_alpha1']*item_1 + self.vars['item_latent']
        print("user shape", user.shape)
        user = tf.nn.l2_normalize(user, axis=1)
        item = tf.nn.l2_normalize(item, axis=1)
        rating = tf.matmul(user, tf.transpose(item))
        return rating,batch_cos, uids, samples



class MyPredictLayerAddMF():
    def __init__(self,placeholders,user,item,tua,name,ac = tf.nn.relu):
        # user = tf.nn.l2_normalize(user, axis=1)
        # item = tf.nn.l2_normalize(item, axis=1)
        user_dim = int(user.shape[0])
        item_dim = int(item.shape[0])
        self.name = name
        self.user = user
        self.item = item
        self.tua = tua
        self.vars = {}
        self.batch_u = placeholders['batch_u']
        self.batch_i = placeholders['batch_i']
        self.batch_j = placeholders['batch_j']
        self.act = ac
        with tf.name_scope(self.name + '_vars'):
            self.vars["user_latent"] = init_variable(user_dim, int(FLAGS.latent_dim), name='user_latent_matrix')
            self.vars["item_latent"] = init_variable(item_dim, int(FLAGS.latent_dim), name='item_latent_matrix')
            # self.vars['item_bias'] = init_variable(item_dim, 1, "item_bias")
            # self.vars['user_alpha1'] = tf.Variable(initial_value=1., name='user_alpha1')
            # self.vars['alpha'] = tf.Variable(initial_value=1., name='alpha')
            # self.vars['alpha2'] = tf.Variable(initial_value=1., name='alpha2')
            # self.vars['user_weight'] = glorot([int(FLAGS.output_dim), int(FLAGS.latent_dim)], name='user_weight')
            # self.vars['item_weight'] = glorot([int(FLAGS.output_dim), int(FLAGS.latent_dim)], name='item_weight')
            # self.vars['user_bias'] = zeros(FLAGS.latent_dim,name='user_bias')
            # self.vars['item_bias'] = zeros(item_dim, name='item_bias')
            # self.vars['emb_projection'] = \
            #     init_variable(int(FLAGS.output_dim), int(FLAGS.output_dim), name='emb_projection_matrix')
            self.vars['user_weight'] = init_variable(int(FLAGS.output_dim), int(FLAGS.latent_dim), name='user_weight')
            self.vars['item_weight'] = init_variable(int(FLAGS.output_dim), int(FLAGS.latent_dim), name='item_weight')
            self.vars['user_bias'] = zeros(int(FLAGS.latent_dim), name='user_bias')
            self.vars['item_bias'] = zeros(int(FLAGS.latent_dim), name='item_bias')




    def __call__(self):
        # user_emb shape: [ user_dim,user_emb_dim]

        # batch_user = tf.nn.embedding_lookup(self.user, self.batch_u)
        # batch_item = tf.nn.embedding_lookup(self.item, self.batch_i)
        # batch_item_j = tf.nn.embedding_lookup(self.item, self.batch_j)
        #
        # batch_user_latent = tf.nn.embedding_lookup(self.vars["user_latent"], self.batch_u)
        # batch_item_latent = tf.nn.embedding_lookup(self.vars["item_latent"], self.batch_i)
        # batch_item_j_latent = tf.nn.embedding_lookup(self.vars["item_latent"], self.batch_j)
        #
        # batch_user_latent = tf.nn.l2_normalize(batch_user_latent, axis=1)
        # batch_item_latent = tf.nn.l2_normalize(batch_item_latent, axis=1)
        # batch_item_j_latent = tf.nn.l2_normalize(batch_item_j_latent, axis=1)
        #
        #
        # pos_cos_mf = get_cosine_similarity(batch_user_latent, batch_item_latent)
        # neg_cos_mf = get_cosine_similarity(batch_user_latent, batch_item_j_latent)

        # pos_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_i)
        # neg_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_j)
        user = tf.matmul(self.user,self.vars['user_weight'])+self.vars['user_bias'] + self.vars['user_latent']
        item = tf.matmul(self.item,self.vars['item_weight'])+self.vars['item_bias'] + self.vars['item_latent']
        l2_user = tf.nn.l2_normalize(user, axis=1)
        l2_item = tf.nn.l2_normalize(item, axis=1)
        batch_user = tf.nn.embedding_lookup(l2_user, self.batch_u)
        batch_item = tf.nn.embedding_lookup(l2_item, self.batch_i)
        batch_item_j = tf.nn.embedding_lookup(l2_item, self.batch_j)


        pos_cos = get_cosine_similarity(batch_user, batch_item)
        neg_cos = get_cosine_similarity(batch_user, batch_item_j)



        # rating = tf.matmul(l2_user,tf.transpose(l2_item)) + self.vars['alpha'] * tf.matmul(l2_user_mf,tf.transpose(l2_item_mf)) + self.vars['item_bias']
        rating = tf.matmul(l2_user,tf.transpose(l2_item))
        print("pos_cos shape", pos_cos.shape)
        print("neg_cos shape", neg_cos.shape)
        return rating,pos_cos,neg_cos






class MyPredictLayer():
    def __init__(self,placeholders,user,item,tua,ac = tf.nn.relu):
        user = tf.nn.l2_normalize(user, axis=1)
        item = tf.nn.l2_normalize(item, axis=1)
        self.user = user
        self.item = item
        self.tua = tua
        self.vars = {}
        self.batch_u = placeholders['batch_u']
        self.batch_i = placeholders['batch_i']
        self.batch_j = placeholders['batch_j']
        #self.reg_batch_u = placeholders['reg_batch_u']
        self.act = ac

    def __call__(self):
        # user_emb shape: [ user_dim,user_emb_dim]

        batch_user = tf.nn.embedding_lookup(self.user, self.batch_u)
        batch_item = tf.nn.embedding_lookup(self.item, self.batch_i)
        batch_item_j = tf.nn.embedding_lookup(self.item, self.batch_j)
      #  reg_batch_u = tf.nn.embedding_lookup(self.item, self.reg_batch_u)
        print("batch_user shape", batch_user.shape)
        print("batch_item shape", batch_item.shape)
        print("batch_item_j shape", batch_item_j.shape)
      #  print("reg_batch_u shape",reg_batch_u.shape)

        pos_cos = get_cosine_similarity(batch_user, batch_item)
        neg_cos = get_cosine_similarity(batch_user, batch_item_j)
        # reg_cos = get_cosine_similarity(batch_item,  batch_item_j)
        rating = tf.matmul(self.user,tf.transpose(self.item))


        print("pos_cos shape", pos_cos.shape)
        print("neg_cos shape", neg_cos.shape)
        # print("reg_cos shape", reg_cos.shape)
        return rating,pos_cos,neg_cos




class MyRateLayer():
    def __init__(self, placeholders, user, item, tau):
        user = tf.nn.l2_normalize(user, axis=1)
        item = tf.nn.l2_normalize(item, axis=1)
        # user = user/tf.expand_dims(tf.reduce_sum(user, axis=1), 1)
        # item = item/tf.expand_dims(tf.reduce_sum(item, axis=1), 1)
        self.user = user
        self.item = item
        self.train_samples = placeholders['train_samples']
        self.uids = placeholders['uids']
        self.name = 'MyRateLayer'
        self.tau = tau
        self.vars = {}
        # self.vars["item_bias"] = tf.expand_dims(zeros(FLAGS.train_samples, name="item_bias"),0)
        with tf.name_scope(self.name + '_vars'):
            self.vars['emb_projection'] = \
                init_variable(int(FLAGS.output_dim), int(FLAGS.output_dim), name='emb_projection_matrix')
            # self.vars['item_bias'] = init_variable(item_dim, 1, "item_bias")
            self.vars['alpha1'] = tf.Variable(initial_value=0., name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1., name='alpha2')


    def __call__(self):

        # Attention
        def attention(ratelayer, inputs, tag, attention_size=32):
            ratelayer.attention_size = attention_size
            ratelayer.tag = tag
            if isinstance(inputs, tuple):
                print("Attention layer - inputs is tuple, concat")
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
                inputs = tf.concat(inputs, 2)

            if ratelayer.time_major:
                # (T,B,D) => (B,T,D)
                inputs = tf.transpose(inputs, [1, 0, 2])

            hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
            print("hidden_size in attention layer", hidden_size)
            print("Att input shape", inputs.shape)

            # Trainable parameters
            with tf.variable_scope('v_' + ratelayer.tag):
                w_omega = tf.get_variable(initializer=tf.random_normal(
                    [hidden_size + FLAGS.latent_dim, ratelayer.attention_size], stddev=0.1), name='w_omega')
                ratelayer.vars['w_omega'] = w_omega
                # b_omega = tf.get_variable(initializer=tf.random_normal(
                #     [ratelayer.attention_size], stddev=0.1), name='b_omega')
                # ratelayer.vars['b_omega'] = b_omega
                u_omega = tf.get_variable(initializer=tf.random_normal(
                    [ratelayer.attention_size], stddev=0.1), name='u_omega')
                ratelayer.vars['u_omega'] = u_omega
                b_v = tf.get_variable(initializer=tf.random_normal([1], stddev=0.1), name='b_v')
                ratelayer.vars['b_v'] = b_v
                # init for projection vars
                ratelayer.vars['project_' + self.tag] = tf.get_variable(
                    initializer=tf.random_normal([FLAGS.latent_dim, FLAGS.latent_dim], stddev=0.1),
                    name='project_' + ratelayer.tag + '_matrix')
                ratelayer.vars['project_bias_' + ratelayer.tag] = tf.get_variable(
                    initializer=tf.random_normal([FLAGS.latent_dim], stddev=0.1),
                    name='b_projection_' + ratelayer.tag)

            # transform and tile
            ratelayer.vars['projected_' + ratelayer.tag + '_latent'] = \
                dot(ratelayer.vars[ratelayer.tag + '_latent'], ratelayer.vars['project_' + ratelayer.tag]) \
                + ratelayer.vars['project_bias_' + ratelayer.tag]
            ratelayer.vars['projected_' + ratelayer.tag + '_latent'] = \
                tf.nn.sigmoid(ratelayer.vars['projected_' + ratelayer.tag + '_latent'])
            projected_latent = tf.tile(
                tf.expand_dims(ratelayer.vars['projected_' + ratelayer.tag + '_latent'], axis=0),
                [inputs.shape[0], 1, 1])

            # concat and non-linear attention additive one like
            # in https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
            v1 = tf.concat([inputs, projected_latent], axis=2)
            v = tf.tanh(tf.tensordot(v1, w_omega, axes=1))
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            print("vu shape", vu.shape)  # vu shape (4, 2005)
            alphas = tf.nn.softmax(vu, name='alphas', axis=0)  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 0)
            return output, alphas

        # self.user, self.vars['alphas_user'] = attention(self, self.user, 'user')
        # self.item, self.vars['alphas_item'] = attention(self, self.item, 'item')



        uids = self.uids
        print("uids shape", uids.shape)
        samples = self.train_samples[:, 1:]
        print("samples shape", samples.shape)
        #user_emb shape: [batch_size, user_dim]
        user_emb = tf.nn.embedding_lookup(self.user, uids)
        #item_emb shape: [batch_size,user_dim, item_dim]
        item_emb = tf.nn.embedding_lookup(self.item, samples)
        print("user_emb shape", user_emb.shape)
        print("item_emb shape", item_emb.shape)

        batch_cos = get_cosine_similarity(user_emb, item_emb)
        # x = tf.nn.l2_normalize(x)
        # y = tf.nn.l2_normalize(y, dim=1)
        # return tf.reduce_sum(tf.multiply(x, y), 1)
        #user_emb shape: [ user_dim,user_emb_dim]
        #item_emb shape: [ item_dim,item_emb_dim]
        usr = tf.nn.l2_normalize(self.user, axis=1)
        item = tf.nn.l2_normalize(self.item, axis=1)
        rating = tf.matmul(usr,tf.transpose(item))


        return rating,batch_cos, uids, samples












class RateLayer():
    def __init__(self, placeholders, user, item, user_dim, item_dim, parentvars, ac=tf.nn.relu):
        self.user = user
        self.item = item
        self.batch_u = placeholders['batch_u']
        self.batch_i = placeholders['batch_i']
        self.batch_j = placeholders['batch_j']
        self.name = 'RateLayer'
        self.ac = ac
        self.vars = {}
        with tf.name_scope(self.name + '_vars'):

            self.vars["user_latent"] = init_variable(user_dim, int(FLAGS.latent_dim), name='user_latent_matrix')
            self.vars["item_latent"] = init_variable(item_dim, int(FLAGS.latent_dim), name='item_latent_matrix')

            # project user emb to item emb space
            self.vars['emb_projection'] = \
                init_variable(int(FLAGS.output_dim), int(FLAGS.output_dim), name='emb_projection_matrix')
            self.vars['item_bias'] = init_variable(item_dim, 1, "item_bias")
            self.vars['alpha1'] = tf.Variable(initial_value=0., name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1., name='alpha2')

    def __call__(self):
        """
        Eq 10 in Attentional Graph Convolutional Networks for Knowledge Concept Recommendation
            in MOOCs in a Heterogeneous View
        """
        # MF
        u_factors = tf.nn.embedding_lookup(self.vars['user_latent'], self.batch_u)
        i_factors = tf.nn.embedding_lookup(self.vars['item_latent'], self.batch_i)
        j_factors = tf.nn.embedding_lookup(self.vars['item_latent'], self.batch_j)
        rate_matrix1_i = tf.reduce_sum(u_factors * i_factors, axis=2)
        rate_matrix1_j = tf.reduce_sum(u_factors * j_factors, axis=2)
        rate_matrix1 = tf.matmul(self.vars['user_latent'], tf.transpose(self.vars['item_latent']))
        print("rate_matrix1 shape:", rate_matrix1.shape)
        # Emb
        u_emb = tf.nn.embedding_lookup(self.user, self.batch_u)
        i_emb = tf.nn.embedding_lookup(self.item, self.batch_i)
        j_emb = tf.nn.embedding_lookup(self.item, self.batch_j)
        u_emb = tf.squeeze(u_emb, axis=1)
        i_emb = tf.squeeze(i_emb, axis=1)
        j_emb = tf.squeeze(j_emb, axis=1)
        u_emb = tf.matmul(u_emb, self.vars['emb_projection'])  # project to item space
        rate_matrix2_i = tf.reduce_sum(u_emb * i_emb, axis=1)
        rate_matrix2_j = tf.reduce_sum(u_emb * j_emb, axis=1)
        projected_user = tf.matmul(self.user, self.vars["emb_projection"])
        rate_matrix2 = tf.matmul(projected_user, tf.transpose(self.item))
        print("rate_matrix2_i shape:", rate_matrix2_i.shape)
        print("rate_matrix2 shape:", rate_matrix2.shape)
        # Bias
        i_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_i)
        j_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_j)
        i_bias = tf.reshape(i_bias, [-1, 1])
        j_bias = tf.reshape(j_bias, [-1, 1])
        # print("i_bias shape:", i_bias.shape)

        # full prediction
        rate_matrix_i = rate_matrix1_i+self.vars['alpha2']*rate_matrix2_i+i_bias
        rate_matrix_j = rate_matrix1_j+self.vars['alpha2']*rate_matrix2_j+j_bias
        #tf.transpose to make the shape (item_size,1) to (1,item_size)
        # +self.vars['alpha2'] * rate_matrix2 + tf.transpose(self.vars['item_bias'])
        rate_matrix = rate_matrix1+ tf.transpose(self.vars['item_bias'])

        # pos-neg diff.
        #why? j is random generated in range(0,item_size) it may be positive or negative, why not use the negative variable definded in m_utils?
        xuij = rate_matrix_i - rate_matrix_j

        # rate_matrix means the prediction score of all u2v pairs
        return rate_matrix, xuij, rate_matrix2, rate_matrix2, tf.transpose(self.vars['item_bias'])


class AttRateLayer():
    """ Combine attention and rating together """
    def __init__(self, placeholders, user, item, user_dim, item_dim, parentvars, ac=tf.nn.relu, time_major=False):
        print("Initializing AttRateLayer")
        self.user = user
        self.item = item
        self.time_major = time_major
        self.batch_u = placeholders['batch_u']
        self.batch_i = placeholders['batch_i']
        self.batch_j = placeholders['batch_j']
        self.name = 'AttRateLayer'
        self.ac = ac
        self.vars = {}
        with tf.name_scope(self.name + '_vars'):
            self.vars["user_latent"] = init_variable(user_dim, int(FLAGS.latent_dim), name='user_latent_matrix')
            self.vars["item_latent"] = init_variable(item_dim, int(FLAGS.latent_dim), name='item_latent_matrix')

            # project user emb to item emb space
            self.vars['emb_projection'] = \
                init_variable(int(FLAGS.output_dim), int(FLAGS.output_dim), name='emb_projection_matrix')
            self.vars['item_bias'] = init_variable(item_dim, 1, "item_bias")
            self.vars['alpha1'] = tf.Variable(initial_value=0., name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1., name='alpha2')

    def __call__(self):
        print("Calling AttRatelayer")
        """
        Eq 10 in Attentional Graph Convolutional Networks for Knowledge Concept Recommendation
            in MOOCs in a Heterogeneous View
        """
        # Attention
        def attention(ratelayer, inputs, tag, attention_size=32):
            ratelayer.attention_size = attention_size
            ratelayer.tag = tag
            if isinstance(inputs, tuple):
                print("Attention layer - inputs is tuple, concat")
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
                inputs = tf.concat(inputs, 2)

            if ratelayer.time_major:
                # (T,B,D) => (B,T,D)
                inputs = tf.transpose(inputs, [1, 0, 2])

            hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
            print("hidden_size in attention layer", hidden_size)
            print("Att input shape", inputs.shape)

            # Trainable parameters
            with tf.variable_scope('v_' + ratelayer.tag):
                w_omega = tf.get_variable(initializer=tf.random_normal(
                    [hidden_size+FLAGS.latent_dim, ratelayer.attention_size], stddev=0.1), name='w_omega')
                ratelayer.vars['w_omega'] = w_omega
                # b_omega = tf.get_variable(initializer=tf.random_normal(
                #     [ratelayer.attention_size], stddev=0.1), name='b_omega')
                # ratelayer.vars['b_omega'] = b_omega
                u_omega = tf.get_variable(initializer=tf.random_normal(
                    [ratelayer.attention_size], stddev=0.1), name='u_omega')
                ratelayer.vars['u_omega'] = u_omega
                b_v = tf.get_variable(initializer=tf.random_normal([1], stddev=0.1), name='b_v')
                ratelayer.vars['b_v'] = b_v
                # init for projection vars
                ratelayer.vars['project_'+self.tag] = tf.get_variable(
                    initializer=tf.random_normal([FLAGS.latent_dim, FLAGS.latent_dim], stddev=0.1),
                    name='project_' + ratelayer.tag+'_matrix')
                ratelayer.vars['project_bias_'+ratelayer.tag] = tf.get_variable(
                    initializer=tf.random_normal([FLAGS.latent_dim], stddev=0.1),
                    name='b_projection_'+ratelayer.tag)

            # transform and tile
            ratelayer.vars['projected_' + ratelayer.tag + '_latent'] = \
                dot(ratelayer.vars[ratelayer.tag + '_latent'], ratelayer.vars['project_' + ratelayer.tag]) \
                + ratelayer.vars['project_bias_' + ratelayer.tag]
            ratelayer.vars['projected_' + ratelayer.tag + '_latent'] = \
                tf.nn.sigmoid(ratelayer.vars['projected_'+ratelayer.tag+'_latent'])
            projected_latent = tf.tile(
                tf.expand_dims(ratelayer.vars['projected_' + ratelayer.tag + '_latent'], axis=0),
                [inputs.shape[0], 1, 1])

            # concat and non-linear attention additive one like
            # in https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
            v1 = tf.concat([inputs, projected_latent], axis=2)
            v = tf.tanh(tf.tensordot(v1, w_omega, axes=1))
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            print("vu shape", vu.shape)  # vu shape (4, 2005)
            alphas = tf.nn.softmax(vu, name='alphas', axis=0)  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 0)
            return output, alphas

        self.user, self.vars['alphas_user'] = attention(self, self.user, 'user')
        self.item, self.vars['alphas_item'] = attention(self, self.item, 'item')

        # MF
        u_factors = tf.nn.embedding_lookup(self.vars['user_latent'], self.batch_u)
        i_factors = tf.nn.embedding_lookup(self.vars['item_latent'], self.batch_i)
        j_factors = tf.nn.embedding_lookup(self.vars['item_latent'], self.batch_j)
        rate_matrix1_i = tf.reduce_sum(u_factors * i_factors, axis=2)
        rate_matrix1_j = tf.reduce_sum(u_factors * j_factors, axis=2)
        rate_matrix1 = tf.matmul(self.vars['user_latent'], tf.transpose(self.vars['item_latent']))
        print("rate_matrix1 shape:", rate_matrix1.shape)
        # Emb
        u_emb = tf.nn.embedding_lookup(self.user, self.batch_u)
        i_emb = tf.nn.embedding_lookup(self.item, self.batch_i)
        j_emb = tf.nn.embedding_lookup(self.item, self.batch_j)
        u_emb = tf.squeeze(u_emb, axis=1)
        i_emb = tf.squeeze(i_emb, axis=1)
        j_emb = tf.squeeze(j_emb, axis=1)
        u_emb = tf.matmul(u_emb, self.vars['emb_projection'])  # project to item space
        rate_matrix2_i = tf.reduce_sum(u_emb * i_emb, axis=1)
        rate_matrix2_j = tf.reduce_sum(u_emb * j_emb, axis=1)
        projected_user = tf.matmul(self.user, self.vars["emb_projection"])
        rate_matrix2 = tf.matmul(projected_user, tf.transpose(self.item))
        print("rate_matrix2_i shape:", rate_matrix2_i.shape)
        print("rate_matrix2 shape:", rate_matrix2.shape)
        # Bias
        i_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_i)
        j_bias = tf.nn.embedding_lookup(self.vars['item_bias'], self.batch_j)
        i_bias = tf.reshape(i_bias, [-1, 1])
        j_bias = tf.reshape(j_bias, [-1, 1])
        # print("i_bias shape:", i_bias.shape)

        # full prediction
        rate_matrix_i = rate_matrix1_i+self.vars['alpha2']*rate_matrix2_i+i_bias
        rate_matrix_j = rate_matrix1_j+self.vars['alpha2']*rate_matrix2_j+j_bias
        rate_matrix = rate_matrix1+self.vars['alpha2']*rate_matrix2+tf.transpose(self.vars['item_bias'])

        # pos-neg diff.
        xuij = rate_matrix_i - rate_matrix_j

        return rate_matrix, xuij, rate_matrix1, rate_matrix2, tf.transpose(self.vars['item_bias'])


class SimpleAttLayer():
    """
    Eq 6 Attentional Graph Convolutional Networks for Knowledge Concept Recommendation
        in MOOCs in a Heterogeneous View
    """
    def __init__(self, attention_size, tag, parentvars, time_major=False):
        print("Initializing SimpleAttLayer - tag:"+tag)
        self.attention_size = attention_size
        self.time_major = time_major
        self.tag = tag
        self.vars = {}

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            print("Attention layer - inputs is tuple, concat")
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if self.time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
        print("hidden_size in attention layer", hidden_size)
        print("Att input shape", inputs.shape)

        # Trainable parameters
        with tf.variable_scope('v_'+self.tag):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            w_omega = tf.get_variable(initializer=tf.random_normal([hidden_size, self.attention_size], stddev=0.1),
                                      name='w_omega')
            self.vars['w_omega'] = w_omega
            b_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='b_omega')
            self.vars['b_omega'] = b_omega
            u_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='u_omega')
            self.vars['u_omega'] = u_omega
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
            print("v shape", v.shape) # v shape (4, 2005, 32)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        print("vu shape", vu.shape) # vu shape (4, 2005)
        alphas = tf.nn.softmax(vu, name='alphas', axis=0)         # (B,T) shape
        self.vars['alphas_'+self.tag] = alphas

        output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 0)

        return output


class linner_layer(Layer):
    def __init__(self, placeholders,tag,inputs_shape,output_shape,act=tf.nn.relu,**kwargs):
        super(linner_layer, self).__init__(**kwargs)
        print("Initializing linner_layer - tag:"+tag)
        self.placeholders = placeholders
        self.tag = tag
        self.act = act
        self.vars = {}
        self.l2_loss = 0
        with tf.name_scope(self.tag + '_vars'):
            self.vars["weight"] = init_variable(inputs_shape, output_shape, name='linner_layer_weight')
            self.vars["bias"] = zeros(output_shape, name='linner_layer_bias')

    def _call(self,inputs):

        output = tf.matmul(inputs, self.vars["weight"]) + self.vars["bias"]
        output = self.act(output)
        return output
