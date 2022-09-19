import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import tensorflow as tf
import tensorflow_compression as tfc
import CNN_img
import MVP_network
from MFE_Net import *

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='./test/CTC/ClassB/')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--f_P", type=int, default=6)
parser.add_argument("--b_P", type=int, default=6)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=123, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--entropy_coding", type=int, default=1)
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()
path1 = args.path
path_list = os.listdir(path1)
for item in path_list:
    args.path = path1 + item + "/"
    print(args.path)
    Height, Width, batch_size,  Channel, activation, \
    GOP_size, GOP_num,path, path_com, path_bin, path_lat = configure(args)
lr_init = 1e-4
batch_size = 1
args.path = path1
Channel = 3
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
    string_mv = tf.squeeze(string_mv, axis=0)
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
    string_res = tf.squeeze(string_res, axis=0)
    res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

    Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC
with tf.variable_scope("MFE_Net"):
    Y1_QE= MFE_Net(Y0_com, Y1_com, batch_size, Height, Width)
        #print("Y1_QE",Y1_QE)
Y1_QE = tf.clip_by_value(Y1_QE,0,1)

fin_loss = L2_loss(Y1_raw,Y1_com)
MVP_loss = L2_loss(flow_tensor,pre_flow)
warp_loss = L2_loss(Y1_raw,Y1_warp)
flow_loss = L2_loss(Y1_raw,Y1_flow)
mc_loss = L2_loss(Y1_raw,Y1_MC)
QE_loss = L2_loss(Y1_raw,Y1_QE)
psnr_QE = 10.0*tf.log(1.0/QE_loss)/tf.log(10.0)
psnr = 10.0*tf.log(1.0/fin_loss)/tf.log(10.0)
psnr_warp= 10.0*tf.log(1.0/warp_loss)/tf.log(10.0)
psnr_mc = 10.0*tf.log(1.0/mc_loss)/tf.log(10.0)
psnr_flow =  10.0*tf.log(1.0/flow_loss)/tf.log(10.0)
saver = tf.train.Saver(max_to_keep=3)
save_path = './myVC_PSNR_' + str(1536)
ckpt = tf.train.get_checkpoint_state(save_path)

saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
print("load  ", ckpt.all_model_checkpoint_paths[-1])
load_iter = int(ckpt.all_model_checkpoint_paths[-1].split("-")[-1])
print(load_iter)

path_list = os.listdir(path1)
for path_item in path_list:
        MV_bpp = []
        res_bpp = []
        qE_psnr = []
        GOP_size = 14
        args.path = path1 + path_item + "/"
        Height, Width, batch_size,  Channel, activation, \
        _, GOP_num,path, path_com, path_bin, path_lat = configure(args)
        for g in range(50):
            frame_index = g * GOP_size + 1
            F0_com = misc.imread(path + 'f' + str(frame_index).zfill(3) + '.png')
            F0_com = np.expand_dims(F0_com, axis=0)
            for f in range(7):
                frame_index = g * GOP_size + 2 + f
                #frame_index = g * GOP_size - f
                F1_raw = misc.imread(path + 'f' + str(frame_index).zfill(3) + '.png')
                F1_raw = np.expand_dims(F1_raw, axis=0)
                if f == 0:
                    Pre_MV = np.random.normal(loc=5.0,scale = 0.2,size = [batch_size,Height,Width,2,3])
                    Pre_state = np.random.normal(loc=0.0,scale = 0.2,size = [2, batch_size, Height//8, Width//8, 64])

                F0_com,string_MV, string_Res,cur_Mv,cur_state,PSNR_QE,= sess.run([Y1_QE,string_mv,string_res,MV,state,psnr_QE],
                                                           feed_dict={Y0_com:F0_com/255.0,
                                                                      Y1_raw:F1_raw/255.0,
                                                                      pre_MV:Pre_MV,
                                                                      pre_state:Pre_state,
                                                                      })
                bpp_Mv = (2 + len(string_MV) ) * 8 / Height / Width
                bpp_Res = (2  + len(string_Res)) * 8 / Height / Width
                F0_com = F0_com * 255.0
                Pre_MV[:,:,:,:,0:2] = Pre_MV[:,:,:,:,1:3]
                Pre_MV[:,:,:,:,2] = cur_Mv
                Pre_state = cur_state
                p_bin = path_bin + '/f' + str(frame_index).zfill(3) + '.bin'
                with open(p_bin, "wb") as ff:
                    ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
                    ff.write(string_MV)
                    ff.write(string_Res)

                misc.imsave(path_com + '/f' + str(frame_index).zfill(3) + '.png', np.float32(np.round(F0_com[0])))
                MV_bpp.append(bpp_Mv)
                res_bpp.append(bpp_Res)
                qE_psnr.append(PSNR_QE)
        all_bit = 0
        bit_list = os.listdir(path_bin)
        for item in bit_list:
            bit_path = path_bin + item
            all_bit += os.path.getsize(bit_path) * 8
        avg_bpp = np.mean(MV_bpp)+np.mean(res_bpp)
        b_mv = np.mean(MV_bpp)
        b_res = np.mean(res_bpp)
        d_qe = np.mean(qE_psnr)
        print(path_item,"Final_PSNR:",d_qe,"avg_bpp:",avg_bpp)

