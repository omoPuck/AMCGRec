import tensorflow as tf
import os
print(tf.__version__)
from models import *
import time
import numpy as np
from scipy import sparse
# from data_utils import *
# from m_utils import *
import logging
import tqdm
import json

# 需要修改的点：每个batch 随机生成negative sample 和对应的 positive sample
# 生成后计算loss



tf.device('/gpu:0')
# ------------------------------------------
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# ------------------------------------------
# Set params
learning_rate = .01
decay_rate = 1
global_steps = 500
decay_steps = 100
neg_samples = 127
reg_num = 10
reg_weight = 0.1
#31 71.77
samples = 1024
batches = 30  # 856067/1024=837

choice_hard_neg_percent=0.1
choice_main_hard_neg_percent=0.01
total_hard_neg_num = 100
hard_neg_start_rank = 100
load_model = False




# print("learning rate:", learning_rate)
# print("global steps:", global_steps)
# print("samples:", samples)
# print("batches", batches)

# ------------------------------------------
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')  # default .5
flags.DEFINE_float('weight_decay', 1e-8, 'Weight for L2 loss on embedding matrix.')  # default 5e-4
flags.DEFINE_integer('output_dim', 64, 'Output_dim of user final embedding.')  # default 64, in paper it seems 100
flags.DEFINE_integer('latent_dim', 30, 'Latent_dim of user&item.')
flags.DEFINE_integer('liner_output_dim', 128, 'Output_dim of linner_layer.')
flags.DEFINE_float('tau', 0.03, 'softmax temperature.')
flags.DEFINE_float('reg_tau', 0.03, 'softmax temperature.')
flags.DEFINE_integer('samples', samples, 'samples.')
flags.DEFINE_float('reg_weight',reg_weight,'Regularization loss weight')
# ------------------------------------------
# L·oad data
support_string_user = ['ucu', 'uvu', 'uctcu', 'uku']
# support_string_user = ['uku']
# support_string_user = ['ucu', 'uvu', 'uctcu']
# support_string_user = ['uku','ucu']
# support_string_item = ['kuk', 'kck']
support_string_item = ['kuk',"kck"]
# support_string_item = ['kuk']
# support_string_user = []
# support_string_item =[]

#UVU在UCU文件夹中 覆盖了！！！ 需要处理

# 测试shape问题
# rating = np.zeros((2005,80802),dtype=np.float32)
# negative = np.zeros((2
# 005,100,2),dtype=np.float32)
# features_user = np.zeros((2005,1000),dtype=np.float32)
# features_item = np.zeros((80802,1000),dtype=np.float32)
model_dir = "./AMCGRec/bestmrr.ckpt".format(neg_samples)
floder_dir = model_dir.replace("/bestmrr.ckpt","")
# save_path = "./output/127neg/"
rating, adjacency_matrix, features_item, features_user, support_user, support_item, negative =load_data(user=support_string_user, item=support_string_item)
if not os.path.exists(floder_dir):
    os.makedirs(floder_dir)

logging.basicConfig(
    level=logging.INFO,
    filename=model_dir.replace("bestmrr.ckpt","") +"train.log",
    filemode="a",
)


logging.info("learning rate:{:4f}".format(learning_rate))
logging.info("samples:{}".format(samples))
logging.info("batches:{}".format(batches))
logging.info("reg_weight:{:4f}".format(reg_weight))
logging.info("neg_samples:{}".format(neg_samples ))
logging.info("total_hard_neg_num:{}".format(total_hard_neg_num))
logging.info("hard_neg_start_rank:{}".format(hard_neg_start_rank))
logging.info("choice_hard_neg_percent:{:4f}".format(choice_hard_neg_percent))
# logging.info("learning rate:", learning_rate)




data_flow = "./data"
# data_flow = "./test/mydata_new/noweightdata"
# data_flow = "./test/mydata/noweightdata"
# rating = np.load(data_flow + "/rating.npy")
# adjacency_matrix = np.load(data_flow + "/adjacency_matrix.npy")
# features_item = np.load(data_flow + "/features_item.npy")
# features_user = np.load(data_flow + "/features_user.npy")
# support_user = np.load(data_flow + "/support_user.npy")
# support_item = np.load(data_flow + "/support_item.npy")
# negative = np.load(data_flow + "/negative.npy")
test_negative = np.load(data_flow + "/test_negative.npy")

# uac_path = "./test/mydata_new/uac_new.json"
# dict
# uac = json.load(open(uac_path))

# print("rating:", rating.shape)

# User size item size
user_dim = rating.shape[0]
item_dim = rating.shape[1]

# Get non-zero indicies  压缩矩阵
straining_matrix = sparse.csr_matrix(rating)

uids, iids = straining_matrix.nonzero()
# 拿到非零的行列  也就是对应的用户和物品下标

print("uids size", len(uids))

# user_support
support_num_user = len(support_string_user)
# item_support
support_num_item = len(support_string_item)
# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32, shape=rating.shape, name="rating"),
    'features_user': tf.placeholder(dtype=tf.float32, shape=features_user.shape, name='features_user'),
    'features_item': tf.placeholder(dtype=tf.float32, shape=features_item.shape, name="features_item"),
    'support_user': [tf.placeholder(dtype=tf.float32, name='support' + str(_)) for _ in range(support_num_user)],
    'support_item': [tf.placeholder(dtype=tf.float32, name='support' + str(_)) for _ in range(support_num_item)],
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative'),
    'batch_u': tf.placeholder(tf.int32, shape=(None, 1), name="user"),
    'batch_i': tf.placeholder(tf.int32, shape=(None, 1), name="item_pos"),
    'batch_j': tf.placeholder(tf.int32, shape=(None, neg_samples), name="item_neg"),
    'reg_batch_u': tf.placeholder(tf.int32, shape=(None, reg_num), name="item_reg"),
    "pos_user":tf.placeholder(tf.int32, shape=(None, 1), name="stru_user_pos"),
    "neg_user":tf.placeholder(tf.int32, shape=(support_num_user, None,total_hard_neg_num), name="stru_user_neg"),
    "pos_item":tf.placeholder(tf.int32, shape=(None, 1), name="stru_item_pos"),
    "neg_item":tf.placeholder(tf.int32, shape=(support_num_item, None,total_hard_neg_num), name="stru_item_neg"),
    "test_negative":tf.placeholder(dtype=tf.int32, shape=test_negative.shape, name='test_negative'),
    # 'train_samples': tf.placeholder(tf.int32, shape=(samples, num_sample + 1), name="batch_sample"),
    'uids': tf.placeholder(tf.int32, shape=(samples, 1), name="user"),
}
global_ = tf.Variable(tf.constant(0))

learning = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

# Create Model
model = AMCGRec(placeholders,
               input_dim_user=features_user.shape[1],
               input_dim_item=features_item.shape[1],
               user_dim=user_dim,
               item_dim=item_dim,
               learning_rate=learning)

# Initialize session

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(floder_dir,sess.graph)

# Init variables
sess.run(tf.global_variables_initializer())

if load_model:
    # Load from previous session
    model.load(model_dir,sess=sess)

# ------------------------------------------
# Train 5.3_model
start_time = time.time()
epoch = 0
mrr_best = 0
hrat5_best = 0
hrat10_best = 0
hrat20_best = 0
ndcgat5_best = 0
ndcgat10_best = 0
ndcgat20_best = 0

# Construct feed dictionary
feed_dict = construct_feed_dict_710(placeholders, features_user, features_item, rating, support_user,
                                support_item, negative,test_negative, FLAGS.dropout)

# total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
# print("Total params of current 5.3_model: {}".format(total_params))


# evaluate test set
# training_matrix_f = './test/mydata_new/uk.csv'
# ground_truth_matrix_f = './test/mydata_new/ground_truth_matrix.npy'
# training_matrix = np.loadtxt(training_matrix_f, delimiter=',')
# ground_truth_matrix = np.load(ground_truth_matrix_f)




origin_user_hard = np.zeros((support_num_user,user_dim,total_hard_neg_num))-1
origin_item_hard = np.zeros((support_num_item,item_dim,total_hard_neg_num))-1



def get_non_zero_hard(k,begin,origin_neg,support):
    mate_len = len(support)
    item_len = len(support[0])
    for i in range(mate_len):
        for j in range(item_len):
            hard_ = np.flip(np.argsort(support[i][j])[-k-begin:-begin])
            for  index,m in enumerate(hard_):
                if support[i][j][m]>0 and index<k  :
                    origin_neg[i,j,index] =  m
                # else:
                #     break
    return origin_neg

user_hard = get_non_zero_hard(total_hard_neg_num,hard_neg_start_rank,origin_user_hard,support_user)
item_hard = get_non_zero_hard(total_hard_neg_num,hard_neg_start_rank,origin_item_hard,support_item)

# def get_stru_neg(pos,negs,thresholds,neg_num):
#     mate_neg = []
#     for i in range(len(thresholds)):
#         i_neg =[]
#         for p in pos:
#             p_negs=[]
#             count =0
#             while count < neg_num:
#                 n_ = np.random.choice(negs[i][p[0]])
#                 if n_ != p[0]:
#                     p_negs.append(n_)
#                     count +=1
#             i_neg.append(p_negs)
#         mate_neg.append(i_neg)
#     return np.array(mate_neg)


def get_stru_neg(pos,hard,neg_num,hard_num,batchj=[]):
    mate_neg = []
    mate_len = len(hard)
    item_len = len(hard[0])
    for i in range(mate_len):
        i_neg = []
        for index, p in enumerate(pos):
            hard_ = np.random.choice(hard[i][p],hard_num)
            non_zero_hard_ = hard_[hard_ > -1]
            if len(batchj) == 0:
                simple_ = np.random.randint(low=0,
                                            high=item_len,
                                            size= neg_num-len(non_zero_hard_))
            else:
                simple_ = np.random.choice(a=batchj[index],
                                           size=neg_num-len(non_zero_hard_)
                                           )
            neg_ = np.concatenate((non_zero_hard_,simple_))
            i_neg.append(neg_)
        mate_neg.append(i_neg)
    return np.array(mate_neg)

def get_hard_batchj(pos,hard,neg_num,hard_num):
    item_len = len(hard)
    batch_neg = []
    for p in pos:
        hard_ = np.random.choice(hard[p], hard_num)
        non_zero_hard_ = hard_[hard_ > -1]
        simple_ = np.random.randint(low=0,
                                    high=item_len,
                                    size=neg_num - len(non_zero_hard_))
        neg_ = np.concatenate((non_zero_hard_, simple_))
        batch_neg.append(neg_)
    return np.array(batch_neg)


out_line = ""

while epoch < global_steps:
    for _ in range(batches):
        idx = np.random.randint(low=0, high=len(uids), size=samples)
        # print("random sample indices:", idx[:10])
        # User batch matching idx
        batch_u = uids[idx].reshape(-1, 1)
        # Pos item
        batch_i = iids[idx].reshape(-1, 1)
        # batch_j=[]
        # # # reg_batch_u = []
        # batch_rating = rating[batch_u.reshape(-1)]
        # # #
        # for i in range(samples):
        #     # n=0
        #     # reg_ = []
        #     neg_idx = np.where(batch_rating[i]==0)[0]
        #     pos_idx = np.where(batch_rating[i]!=0)[0]
        #     neg_ = np.random.choice(neg_idx,neg_samples)
        #     # while(n < reg_num):
        #     #     pos = np.random.choice(pos_idx)
        #     #     if(pos != batch_u[i]):
        #     #         reg_.append(pos)
        #     #         n = n + 1
        #     # reg_ = np.array(reg_)
        #     # reg_batch_u.append(reg_)
        #     batch_j.append(neg_)
        # batch_j = np.array(batch_j)
        # reg_batch_u = np.array(reg_batch_u)
        batch_j = np.random.randint(
            low=0,
            high=item_dim,
            size=(samples, neg_samples),
            dtype="int32"
        )
        # batch_j = get_hard_batchj(pos=batch_i.reshape(-1),hard=item_hard[0],neg_num=neg_samples,hard_num=int(neg_samples * choice_main_hard_neg_percent))
        batch_u_pos = batch_u
        # batch_u_neg = np.random.randint(
        #     low=0,
        #     high=user_dim,
        #     size=(support_num_user, samples, total_hard_neg_num),
        #     dtype="int32"
        # )
        batch_u_neg = get_stru_neg(pos=batch_u.reshape(-1),hard=user_hard,neg_num=total_hard_neg_num,hard_num=int(total_hard_neg_num * choice_hard_neg_percent))
        batch_i_pos = batch_i
        batch_i_neg = get_stru_neg(pos=batch_i.reshape(-1),hard=item_hard,neg_num=total_hard_neg_num,hard_num=int(total_hard_neg_num * choice_hard_neg_percent))
        # batch_i_neg = np.random.randint(
        #     low=0,
        #     high=item_dim,
        #     size=(support_num_item, samples, total_hard_neg_num),
        #     dtype="int32"
        # )

        # print("batchjshape{}".format(batch_j.shape))
        batch_u = batch_u.astype("float32")
        batch_i = batch_i.astype("float32")
        batch_j = batch_j.astype("float32")
        batch_u_neg = batch_u_neg.astype("float32")
        batch_u_pos = batch_u_pos.astype("float32")
        batch_i_neg = batch_i_neg.astype("float32")
        batch_i_pos = batch_i_pos.astype("float32")
        # # reg_batch_u = reg_batch_u.astype("float32")

        # print("batch_j.shape:", batch_j.shape)

        # Get user index shape [samples, 1]
        # sample表示每个batch有多少个正样本
        # print("uids:", uids)

        # For each user,get sample item index
        # negative_samples表示每个正样本有多少个负样本进行训练
        # print("train_sample:", train_sample)
        # print("train_sample:", train_sample.shape)
        # # Neg item
        # # 生成1024个 比item_dim小的数？
        # #J 应该是1024个不在uac中的item下标 这边应该有问题？
        # batch_j = np.random.randint(
        #     low=0,
        #     high=item_dim,
        #     size=(samples, 1),
        #     dtype="int32"
        # )
        # # To feed, need to change dtype
        # # batch_u = user
        # batch_u = batch_u.astype("float32")
        # batch_i = batch_i.astype("float32")
        # batch_j = batch_j.astype("float32")

        # Update feed_dict
        # feed_dict.update({placeholders['features_user']: features_user})
        # feed_dict.update({placeholders['features_item']: features_item})
        feed_dict.update({placeholders['batch_u']: batch_u})
        feed_dict.update({placeholders['batch_i']: batch_i})
        feed_dict.update({placeholders['batch_j']: batch_j})
        feed_dict.update({placeholders['pos_user']:batch_u_pos})
        feed_dict.update({placeholders['neg_user']:batch_u_neg})
        feed_dict.update({placeholders['pos_item']:batch_i_pos})
        feed_dict.update({placeholders['neg_item']:batch_i_neg})

        #        feed_dict.update({placeholders['reg_batch_u']: reg_batch_u})
        # feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # print(placeholders['uids'])
        feed_dict.update({global_: epoch})

        # Train with batch
        # rate1, rate2, bias, alphas_user, alphas_item
        # , user_alpha1, item_alpha1

        _,los, l2_los,Struc_los,HR1, HR5, HR10, HR20, NDCG5, NDCG10, NDCG20, MRR, user, item, result, \
        rate_matrix ,t_HR1, t_HR5, t_HR10, t_HR20, t_NDCG5, t_NDCG10, t_NDCG20, t_MRR = \
            sess.run([model.train_op,model.los, model.l2_loss,model.struc_los,
                      # 5.3_model.layers[-1].vars['alpha1'], 5.3_model.layers[-1].vars['alpha2'],
                      model.hrat1,
                      model.hrat5, model.hrat10, model.hrat20,
                      model.ndcg5, model.ndcg10, model.ndcg20,
                      model.mrr,
                      model.user, model.item,
                      model.result, model.rate_matrix,
                      model.test_hrat1,
                      model.test_hrat5, model.test_hrat10, model.test_hrat20,
                      model.test_ndcg5, model.test_ndcg10, model.test_ndcg20,
                      model.test_mrr
                      # model.layers[-1].vars["item_alpha1"],
                      ],
                     feed_dict)
    AUC,t_AUC= sess.run([model.auc,model.test_auc],feed_dict)
    if epoch % 1 == 0:
        Line =  "epoch:"+str(epoch) +" Train:"   + \
                " Total-Loss:{:8.6f}".format(los) + \
                " L2-Loss:{:8.6f}".format(l2_los) + \
                " Model-Loss:{:8.6f}".format(los - l2_los - Struc_los) + \
                " Struc-loss:{:8.6f}".format(Struc_los) + \
                " HR@1:{:8.6f}".format(HR1) + \
                " HR@5:{:8.6f}".format(HR5) + \
                " HR@10:{:8.6f}".format(HR10) + \
                " HR@20:{:8.6f}".format(HR20) + \
                " nDCG@5:{:8.6f}".format(NDCG5) + \
                " nDCG@10:{:8.6f}".format(NDCG10) + \
                " nDCG@20:{:8.6f}".format(NDCG20) + \
                " MRR:{:8.6f}".format(MRR) + \
                " AUC:{:8.6f}".format(AUC) + \
                " Test:"  + \
                " HR@1:{:8.6f}".format(t_HR1) + \
                " HR@5:{:8.6f}".format(t_HR5) + \
                " HR@10:{:8.6f}".format(t_HR10) + \
                " HR@20:{:8.6f}".format(t_HR20) + \
                " nDCG@5:{:8.6f}".format(t_NDCG5) + \
                " nDCG@10:{:8.6f}".format(t_NDCG10) + \
                " nDCG@20:{:8.6f}".format(t_NDCG20) + \
                " MRR:{:8.6f}".format(t_MRR) +\
                " AUC:{:8.6f}".format(t_AUC)
        # "item_alpha1:{:8.6f}".format(item_alpha1)



        out_line = time.ctime() + " {:10.2f}s passed  ".format(time.time() - start_time) + Line
        print(out_line)
        summary = tf.Summary(value=[tf.Summary.Value(tag="main_loss", simple_value=los),tf.Summary.Value(tag="struc_los",simple_value = Struc_los),tf.Summary.Value(tag="thr5",simple_value = t_HR5)
                                    ,tf.Summary.Value(tag="thr10",simple_value = t_HR10),tf.Summary.Value(tag="thr20",simple_value = t_HR20)
                                    ,tf.Summary.Value(tag="tndgc5",simple_value = t_NDCG5),tf.Summary.Value(tag="tndgc10",simple_value = t_NDCG10),tf.Summary.Value(tag="tndgc20",simple_value = t_NDCG20)
                                    ,tf.Summary.Value(tag="tmrr",simple_value = t_MRR),tf.Summary.Value(tag="mrr",simple_value = MRR),tf.Summary.Value(tag="tauc",simple_value = t_AUC)])

        writer.add_summary(summary, epoch)
        # writer.add_summary(t_HR5, epoch)
        # out_line = Line + T_line

        logging.info(out_line)



    epoch += 1

    # Save rating prediction with best performance
    if epoch > 100:
        # tmrr = t_MRR / sample_num
        # if HR5 > hrat5_best:
        #     print("Best HR5-{} updated at epoch:{}".format(HR5, epoch))
        #     hrat5_best = HR5
        #     with open('./output/m_rating_pred_besthr5.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="besthr5")
        #     np.save('./output/alphas_user_besthr5', alphas_user)
        #     np.save('./output/alphas_item_besthr5', alphas_user)
        # if HR10 > hrat10_best:
        #     print("Best HR10-{} updated at epoch:{}".format(HR10, epoch))
        #     hrat10_best = HR10
        #     with open('./output/m_rating_pred_besthr10.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="besthr10")
        #     np.save('./output/alphas_user_besthr10', alphas_user)
        #     np.save('./output/alphas_item_besthr10', alphas_user)
        # if HR20 > hrat20_best:
        #     print("Best HR20-{} updated at epoch:{}".format(HR20, epoch))
        #     hrat20_best = HR20
        #     with open('./output/m_rating_pred_besthr20.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="besthr20")
        #     np.save('./output/alphas_user_besthr20', alphas_user)
        #     np.save('./output/alphas_item_besthr20', alphas_user)
        # if NDCG5 > ndcgat5_best:
        #     print("Best NDCG5-{} updated at epoch:{}".format(NDCG5, epoch))
        #     ndcgat5_best = NDCG5
        #     with open('./output/m_rating_pred_bestndcg5.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="bestndcg5")
        #     np.save('./output/alphas_user_bestndcg5', alphas_user)
        #     np.save('./output/alphas_item_bestndcg5', alphas_user)
        # if NDCG10 > ndcgat10_best:
        #     print("Best NDCG10-{} updated at epoch:{}".format(NDCG10, epoch))
        #     ndcgat10_best = NDCG10
        #     with open('./output/m_rating_pred_bestndcg10.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="bestndcg10")
        #     np.save('./output/alphas_user_bestndcg10', alphas_user)
        #     np.save('./output/alphas_item_bestndcg10', alphas_user)
        # if NDCG20 > ndcgat20_best:
        #     print("Best NDCG20-{} updated at epoch:{}".format(NDCG20, epoch))
        #     ndcgat20_best = NDCG20
        #     with open('./output/m_rating_pred_bestndcg20.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     5.3_model.save(sess, info="bestndcg20")
        #     np.save('./output/alphas_user_bestndcg20', alphas_user)
        #     np.save('./output/alphas_item_bestndcg20', alphas_user)
        # if tmrr > mrr_best:
        if  MRR > mrr_best:
            # print("Best MRR-{} updated at epoch:{}".format(tmrr, epoch))
            print("Best MRR-{} updated at epoch:{}".format(MRR, epoch))
            # mrr_best = tmrr

            logging.info("Best MRR-{} updated at epoch:{}".format(MRR, epoch))
            mrr_best = MRR

            if not os.path.exists(floder_dir):
                os.makedirs(floder_dir)
                with open(floder_dir + "/rating.p", 'wb') as f:
                    pkl.dump(rate_matrix, f)


                # np.savetxt(floder_dir + "/rating.csv", rate_matrix, delimiter=",")
            else:
                with open(floder_dir+"/rating.p", 'wb') as f:
                    pkl.dump(rate_matrix, f)
                # np.savetxt(floder_dir + "/rating.csv", rate_matrix, delimiter=",")
            # Save
            model.save(model_dir,sess=sess, info="bestmrr")
            # np.save('./output/alphas_user_mrr', alphas_user)
            # np.save('./output/alphas_item_mrr', alphas_user)


    # Save rating prediction every 50 epoch
    # if (epoch) % 50 == 0:
    #     with open('./output/m_rating_pred_ep{}.p'.format(epoch-1), 'wb') as f:
    #         pkl.dump(rate_matrix, f)
    #     # Save
    #     5.3_model.save(sess, info="ep{}".format(epoch))
