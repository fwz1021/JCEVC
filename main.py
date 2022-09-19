import argparse
import tensorflow as tf
import tensorflow_compression as tfc
import CNN_img
import load
import gc
import MVP_network
import time
from matplotlib import pyplot as plt
from MFE_Net import *
os.environ['CUDA_VISIBLE_DEVICES']='4'
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--l", type=int, default=1536, choices=[256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

batch_size = 1
Height = 240
Width = 256
Channel = 3
lr_init = 1e-4

folder = np.load('folder.npy')
Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
pre_MV = tf.placeholder(tf.float32,[batch_size,Height,Width,2,3])
learning_rate = tf.placeholder(tf.float32, [])
pre_state = tf.placeholder(tf.float32, [2, batch_size, Height//8, Width//8, 64])
state_c,state_h = tf.split(pre_state,2,axis = 0)

with tf.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
Y1_flow = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)


with tf.variable_scope("MVP_Net"):
    pre_flow = MVP_network.MVP_net(pre_MV)
res_flow = flow_tensor - pre_flow

with tf.variable_scope("res_flow_Net"):
    flow_latent = CNN_img.MV_analysis(res_flow, args.N, args.M)
    entropy_bottleneck_mv = tfc.EntropyBottleneck()
    string_mv = entropy_bottleneck_mv.compress(flow_latent)
    #string_mv = tf.squeeze(string_mv, axis=0)
    flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=True)

    flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
MV = flow_hat + pre_flow
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, MV)
with tf.variable_scope('MC_net'):
    Y1_MC,state_c,state_h = MC_Generator_1(Y1_warp=Y1_warp,Y0_com=Y0_com,state_c =state_c ,state_h = state_h,Height=Height, Width = Width)
    state  = tf.stack([state_c,state_h],axis=0)

with tf.variable_scope('discriminator', reuse=False):
    real_out, real_cam = discriminator(Y1_raw,"raw",sn=True)
    fake_out, fake_cam = discriminator(Y1_MC,"MC",sn=True)

Y1_res = Y1_raw - Y1_MC
with tf.variable_scope("res_Net"):
    res_latent = CNN_img.Res_analysis(Y1_res, num_filters=args.N, M=args.M)

    entropy_bottleneck_res = tfc.EntropyBottleneck()
    string_res = entropy_bottleneck_res.compress(res_latent)
    # string_res = tf.squeeze(string_res, axis=0)
    res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

    Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)
# Reconstructed frame
Y1_com = Res_hat + Y1_MC

with tf.variable_scope("MFE_Net"):
    Y1_QE = MFE_Net(Y0_com, Y1_com, batch_size, Height, Width)

# Total number of bits divided by number of pixels.
train_bpp_MV = tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

#discriminator loss
D_loss = discriminator_loss(real_out,fake_out) + discriminator_loss(real_cam,fake_cam)
t_vars = tf.trainable_variables()    #返回所有参数

D_vars = [var for var in t_vars if 'discriminator' in var.name]
mvp_vars = [var for var in t_vars if "MVP_Net" in var.name]
res_flow_vars = [var for var in t_vars if "res_flow_Net" in var.name]
res_vars = [var for var in t_vars if "res_Net" in var.name]
mc_vars = [var for var in t_vars if "MC_net" in var.name]
mfe_vars = [var for var in t_vars if "MFE_Net" in var.name]
glob_vars = [var for var in t_vars if "flow_motion" not in var.name]
#generator loss
G_loss = generator_loss(Y1_raw,Y1_MC,fake_out,fake_cam)

fin_loss = L2_loss(Y1_raw,Y1_com)
MVP_loss = L2_loss(flow_tensor,pre_flow)
warp_loss = L2_loss(Y1_raw,Y1_warp)
res_loss = L2_loss(Y1_res,Res_hat)
mv_loss = L2_loss(flow_hat,res_flow)
mfe_loss = L2_loss(Y1_QE,Y1_raw)

res_flow_v = tf.reduce_mean(tf.square(res_flow))
res_v = tf.reduce_mean(tf.square(Y1_res))
# The rate-distortion cost.
l = args.l

train_loss_total = l * mfe_loss + (train_bpp_MV + train_bpp_Res)
train_loss_MV = l * warp_loss + train_bpp_MV
train_loss_MVP = l * MVP_loss
train_loss_MC = l * G_loss + train_bpp_MV
train_loss_res_flow = l * mv_loss + train_bpp_MV
train_loss_RES = l * res_loss + train_bpp_Res
train_loss_MFE = l * mfe_loss

step = tf.train.create_global_step()

train_op_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, global_step=step,var_list=D_vars)
train_MC = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MC, global_step=step,var_list=mc_vars)
train_MVP = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MVP, global_step=step,var_list=mvp_vars)
train_MV = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MV, global_step=step,var_list=res_flow_vars)
train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total, global_step=step,var_list=glob_vars)
train_res_flow = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_res_flow, global_step=step,var_list=res_flow_vars)
train_RES = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_RES, global_step=step,var_list=res_vars)
train_op_MFE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MFE, global_step=step,var_list=mfe_vars)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

#用于创造一个操作
train_op_MV = tf.group(train_MV, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MC = tf.group(train_MC, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MVP = tf.group(train_MVP, aux_step, entropy_bottleneck_mv.updates[0])
train_op_res_flow = tf.group(train_res_flow, aux_step, entropy_bottleneck_mv.updates[0])
train_op_RES = tf.group(train_RES, aux_step2, entropy_bottleneck_res.updates[0])

train_op_all = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])
#PSNR
mc_loss = L2_loss(Y1_raw,Y1_MC)
flow_loss = L2_loss(Y1_raw,Y1_flow)
psnr_flow = 10.0*tf.log(1.0/flow_loss)/tf.log(10.0)
psnr = 10.0*tf.log(1.0/fin_loss)/tf.log(10.0)
psnr_warp= 10.0*tf.log(1.0/warp_loss)/tf.log(10.0)
psnr_mc = 10.0*tf.log(1.0/res_v)/tf.log(10.0)
psnr_QE = 10.0*tf.log(1.0/mfe_loss)/tf.log(10.0)

tf.summary.scalar('psnr', psnr)
tf.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
save_path = './myVC_PSNR_' + str(l)
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
saver = tf.train.Saver(max_to_keep=5)
ckpt = tf.train.get_checkpoint_state(save_path)

sess.run(tf.global_variables_initializer())
var_motion = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
saver_motion = tf.train.Saver(var_list=var_motion, max_to_keep=None)
saver_motion.restore(sess, save_path='motion_flow/model.ckpt-200000')

iter = 0
flag = 1
forward_flag = 0
BPP_MV = []
BPP_Res = []
while(True):
    # load model
    frame = 7
    lr = lr_init
    if ckpt and flag:
        flag = 0
        print("........load model........")
        print("load  ", ckpt.all_model_checkpoint_paths[-1])
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
        iter = int(ckpt.all_model_checkpoint_paths[-1].split('-')[-1])
        print("success")
    start = time.time()
    if iter <= 40000 or (iter >=300000 and iter < 330000):
        train_op = train_op_MVP
    elif iter <= 80000 or (iter >=330000 and iter < 360000):
        train_op = train_op_MV
    elif iter <= 160000 or (iter >=360000 and iter < 400000):
        train_op = train_op_MC
    elif iter <= 200000 or (iter >=400000 and iter < 430000):
        train_op = train_op_RES
    elif iter <= 280000 or (iter >=430000 and iter < 470000):
        train_op = train_op_MFE
    else:
        train_op = train_op_all

    if iter <= 300000:
        lr = lr_init
    elif iter <= 600000:
        lr = lr_init/10.0
    else:
        lr = lr_init/100.0
    forward_flag += 1
    data = np.zeros([frame,batch_size,Height,Width,Channel])
    data = load.load_data(data,frame,batch_size,Height,Width,Channel,folder,forward_flag)
    Pre_MV = np.random.normal(loc=5.0,scale = 0.2,size = [batch_size,Height,Width,2,3])
    Pre_state = np.random.normal(loc=0.0,scale = 0.2,size = [2, batch_size, Height//8, Width//8, 64])
    print("forward_flag",forward_flag%2)
    for f in range(frame-1):
        if f == 0:
            F0_com = data[0]
            F1_raw = data[1]
            _,F1_decode,cur_MV,cur_state,final_PSNR,psnr_Warp,bpp_MV,bpp_Res,MVP_error,flow_PSNR,res_mv_error,RES_flow,MC_PSNR,PSNR_QE\
                = sess.run([train_op,Y1_QE,MV,state,psnr,psnr_warp,train_bpp_MV ,train_bpp_Res,MVP_loss,psnr_flow,mv_loss,res_flow_v,psnr_mc,psnr_QE],feed_dict={
                                                   Y0_com:F0_com/255.0,
                                                   Y1_raw:F1_raw/255.0,
                                                   pre_MV:Pre_MV,
                                                   pre_state:Pre_state,
                                                   learning_rate:lr})

            Pre_MV[:,:,:,:,0:2] = Pre_MV[:,:,:,:,1:3]
            Pre_MV[:,:,:,:,2] = cur_MV
            Pre_state = cur_state
            if iter < 260000:
                F1_decode = F1_raw/255.0
            if iter >=120000 and iter <=220000:
                _,d_loss = sess.run([train_op_D,D_loss],
                                    feed_dict={Y0_com: F0_com/255.0,
                                               Y1_raw: F1_raw/255.0,
                                               pre_MV: Pre_MV,
                                               pre_state: Pre_state,
                                               learning_rate: lr
                                               })
        else:
            F0_com = F1_decode*255.0
            F1_raw = data[f+1]
            _,F1_decode,cur_MV,cur_state,final_PSNR,psnr_Warp,bpp_MV,bpp_Res,MVP_error,flow_PSNR,res_mv_error,RES_flow,MC_PSNR,PSNR_QE\
                = sess.run([train_op,Y1_QE,MV,state,psnr,psnr_warp,train_bpp_MV ,train_bpp_Res,MVP_loss,psnr_flow,mv_loss,res_flow_v,psnr_mc,psnr_QE],feed_dict={
                                                   Y0_com:F0_com/255.0,
                                                   Y1_raw:F1_raw/255.0,
                                                   pre_MV:Pre_MV,
                                                   pre_state:Pre_state,
                                                   learning_rate:lr})
            Pre_MV[:,:,:,:,0:2] = Pre_MV[:,:,:,:,1:3]
            Pre_MV[:,:,:,:,2] = cur_MV
            Pre_state = cur_state
            if iter < 300000:
                F1_decode = F1_raw/255.0
            # update D

            if iter >=120000 and iter <=220000:
                _,d_loss = sess.run([train_op_D,D_loss],
                                    feed_dict={Y0_com: F0_com/255.0,
                                               Y1_raw: F1_raw/255.0,
                                               pre_MV: Pre_MV,
                                               pre_state: Pre_state,
                                               learning_rate: lr
                                               })
        print("iter:", iter,"PSNR_QE:%.2f"%PSNR_QE,"final_PSNR:%.2f"%final_PSNR,"MC_PSNR:%.2f"%MC_PSNR,"psnr_Warp:%.2f"%psnr_Warp,"flow_PSNR:%.2f"%flow_PSNR,"bpp_MV:%.3f"%bpp_MV,"bpp_Res:%.3f"%bpp_Res,"MVP_error:%.2f"%MVP_error,"res_mv_error:%.2f"%res_mv_error,"RES_flow:%.2f"%RES_flow,)
        BPP_MV.append(bpp_MV)
        BPP_Res.append(bpp_Res)
        iter = iter + 1
        if iter % 500 == 0:
             merged_summary_op = tf.summary.merge_all()
             summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/255.0,
                                                                  Y1_raw: F1_raw/255.0,
                                                                  pre_MV: Pre_MV,
                                                                  pre_state: Pre_state,
                                                                  learning_rate: lr})

             summary_writer.add_summary(summary_str, iter)

        if iter % 10000 == 0:
             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 900000:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decode

    gc.collect()










