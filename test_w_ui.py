import os
import json 

import streamlit as st
from tgcn_model import GCN_muti_att
from configs import Config
import torch
import numpy as np
import utils 
from sign_dataset_ui import Sign_Dataset
from train_utils import train, validation
from preprocess import video_to_frames
from sklearn.preprocessing import LabelEncoder 

#Function to save uploaded file
def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempvid.mp4"), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("SavedFile")


st.title("Interpreting Sign Language")
st.markdown("Welcome to our Deep Learning Project \
that seeks to help you in converting short videos \
in sign language into the words \
simply upload an MP4 file and watch the magic happen :)")
##btn = st.button("Term 7 is Over!")
##
##if btn:
##    st.balloons()


# change root and subset accordingly.
root = '/home/jovyan/Documents/DL/DL_Project/WLASL'
trained_on = 'asl100'
trained_on_pose = 'asl100'

checkpoint = 'ckpt.pth'

split_file = os.path.join(root, 'data/splits/{}.json'.format(trained_on_pose))
pose_data_root = os.path.join(root, 'data/pose_per_individual_videos')
config_file = os.path.join(root, 'code/TGCN/configs/{}.ini'.format(trained_on))
configs = Config(config_file)

num_samples = configs.num_samples
hidden_size = configs.hidden_size
drop_p = configs.drop_p
num_stages = configs.num_stages
batch_size = configs.batch_size
#batch_size = 1

###set up the dataset
##train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
##                            img_transforms=None, video_transforms=None, num_samples=num_samples)
##
### setup the model
###model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=hidden_size,
##                    #num_class=int(trained_on[3:]), p_dropout=drop_p, num_stage=num_stages).cuda()
##
##model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
##                    num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cuda()
##
##
###name of the model, change accordingly
##checkpoint = 'gcn_epoch=6_val_acc=0.12015503875968993.pth'
##
##checkpoint = torch.load(os.path.join(root, 'code/TGCN/checkpoints/{}/{}'.format(trained_on, checkpoint)))
##model.load_state_dict(checkpoint)
##model.eval()
##
##f = st.file_uploader("Upload a Video", type = ['mp4'])
##if not f:
##    st.empty().info("Please upload a file of type: mp4")
##    
###transform the mp4 file
##if f:
##    save_uploadedfile(f)
##    content = video_to_frames("tempvid.mp4")
##    st.write(content)
##    model(content)
#set up the dataset
dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                       img_transforms=None, video_transforms=None,
                       num_samples=num_samples,
                       sample_strategy='k_copies',
                       test_index_file=split_file
                       )
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# setup the model
#model = GCN_muti_att(input_feature=num_samples * 2, hidden_feature=hidden_size,
                    #num_class=int(trained_on[3:]), p_dropout=drop_p, num_stage=num_stages).cuda()

model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=num_samples*2,
                    num_class=len(dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cuda()


#name of the model, change accordingly
checkpoint = 'gcn_epoch=6_val_acc=0.12015503875968993.pth'

checkpoint = torch.load(os.path.join(root, 'code/TGCN/checkpoints/{}/{}'.format(trained_on, checkpoint)))
model.load_state_dict(checkpoint)
model.eval()


f = st.file_uploader("Upload a Video", type = ['mp4'])
if not f:
    st.empty().info("Please upload a file of type: mp4")
    
#transform the mp4 file
if f:
    
##    X, y, video_ids = dataset[0]
##    st.write(X.size(), y)
##    #X, y = X.cuda(), y.cuda().view(-1, )
##
##    all_output = []
##
##    model.cpu()
##    output = model(X)
##    all_output.append(output)
##
##    all_output = torch.stack(all_output, dim=1)
##    output = torch.mean(all_output, dim=1)

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # distribute data to device
            X, y, video_ids = data
            #Prints the first video id 
            st.write(video_ids[0])
            X, y = X.cuda(), y.cuda().view(-1, )

            all_output = []

            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i + 1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            #y_pred = output.max(1, keepdim=True)[1]
            y_out = output.argmax(1, keepdim =True)[1].cpu()
            y_out = dataset.label_encoder.inverse_transform(y_out).tolist()
            st.write('The predicted output is: ', y_out[0])
            break
        st.write('Was the output correct? ')
if st.button('Yes'):
    st.write('We hope you enjoyed using our platform')
elif st.button('No'):
    st.write('Thank you for your feedback, we hope to serve you better next time!')


##    save_uploadedfile(f)
##    content = video_to_frames("tempvid.mp4")
##    st.write(content)    

    
