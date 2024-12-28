import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import random as rd
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import gc

sns.set_style('darkgrid')

# Set the title of the Streamlit app
st.title("Detecting Anomalies in Financial Transactions")

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")

# Display versions and configuration info
st.write(f"[LOG {now}] The CUDNN backend version: {torch.backends.cudnn.version()}")
st.write(f"[LOG {now}] The Python version: {sys.version}")
st.write(f"[LOG {now}] The PyTorch version: {torch.__version__}")

# Initialize deterministic seed
seed_value = 1234
rd.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if USE_CUDA:
    torch.cuda.manual_seed(seed_value)

# Load dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    ori_dataset = pd.read_csv(uploaded_file)
    st.write(f"[LOG {now}] Transactional dataset of {ori_dataset.shape[0]} rows and {ori_dataset.shape[1]} columns loaded")
    st.dataframe(ori_dataset.head(10))

    label = ori_dataset.pop('label')

    # Plot categorical attributes
    st.subheader("Categorical Attributes Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.countplot(x=ori_dataset['BSCHL'], ax=ax[0])
    ax[0].set_title('Distribution of BSCHL attribute values')
    sns.countplot(x=ori_dataset['HKONT'], ax=ax[1])
    ax[1].set_title('Distribution of HKONT attribute values')
    st.pyplot(fig)

    # Transform categorical attributes
    categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'WAERS', 'BUKRS']
    ori_dataset_categ_transformed = pd.get_dummies(ori_dataset[categorical_attr_names])

    # Plot numeric attributes
    st.subheader("Numeric Attributes Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    sns.displot(ori_dataset['DMBTR'], ax=ax[0])
    ax[0].set_title('Distribution of DMBTR amount values')
    sns.displot(ori_dataset['WRBTR'], ax=ax[1])
    ax[1].set_title('Distribution of WRBTR amount values')
    st.pyplot(fig)

    # Transform numeric attributes
    numeric_attr_names = ['DMBTR', 'WRBTR']
    numeric_attr = ori_dataset[numeric_attr_names] + 1e-7
    numeric_attr = numeric_attr.apply(np.log)
    ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

    # Pairplot of numeric attributes
    numeric_attr_vis = ori_dataset_numeric_attr.copy()
    numeric_attr_vis['label'] = label
    g = sns.pairplot(data=numeric_attr_vis, vars=numeric_attr_names, hue='label')
    g.fig.suptitle('Distribution of DMBTR vs. WRBTR amount values')
    g.fig.set_size_inches(15, 5)
    st.pyplot(g)

    # Combine transformed data
    ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis=1)

    # Encoder model definition
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.encoder_L1 = nn.Linear(in_features=ori_subset_transformed.shape[1], out_features=3, bias=True)
            nn.init.xavier_uniform_(self.encoder_L1.weight)
            self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        def forward(self, x):
            x = self.encoder_R1(self.encoder_L1(x))
            return x

    # Decoder model definition
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.decoder_L1 = nn.Linear(in_features=3, out_features=ori_subset_transformed.shape[1], bias=True)
            nn.init.xavier_uniform_(self.decoder_L1.weight)
            self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        def forward(self, x):
            x = self.decoder_R1(self.decoder_L1(x))
            return x

    encoder_train = Encoder()
    decoder_train = Decoder()
    if USE_CUDA:
        encoder_train = encoder_train.cuda()
        decoder_train = decoder_train.cuda()

    st.write(f"[LOG {now}] Encoder architecture:\n\n{encoder_train}")
    st.write(f"[LOG {now}] Decoder architecture:\n\n{decoder_train}")

    # Training
    learning_rate = 0.001
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)
    num_epochs = 5
    mini_batch_size = 128

    ori_subset_transformed = ori_subset_transformed.apply(pd.to_numeric, errors='coerce').dropna()
    ori_subset_transformed = ori_subset_transformed.astype('float64')
    torch_dataset = torch.from_numpy(ori_subset_transformed.values).float()
    if USE_CUDA:
        torch_dataset = torch_dataset.cuda()
    dataloader = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=True)

    losses = []
    data = autograd.Variable(torch_dataset)

    for epoch in range(num_epochs):
        mini_batch_count = 0
        if USE_CUDA:
            encoder_train.cuda()
            decoder_train.cuda()
        encoder_train.train()
        decoder_train.train()
        start_time = datetime.now()

        for mini_batch_data in dataloader:
            mini_batch_count += 1
            mini_batch_torch = autograd.Variable(mini_batch_data)
            z_representation = encoder_train(mini_batch_torch)
            mini_batch_reconstruction = decoder_train(z_representation)
            reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_torch)
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            reconstruction_loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()

        encoder_train.cpu().eval()
        decoder_train.cpu().eval()
        reconstruction = decoder_train(encoder_train(data))
        reconstruction_loss_all = loss_function(reconstruction, data)
        losses.append(reconstruction_loss_all.item())
        st.write(f"[LOG {now}] Training status, epoch: [{epoch + 1}/{num_epochs}], loss: {reconstruction_loss_all.item()}")

        # Save models
        save_dir = "models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(encoder_train.state_dict(), os.path.join(save_dir, f"ep_{epoch + 1}_encoder_model.pth"))
        torch.save(decoder_train.state_dict(), os.path.join(save_dir, f"ep_{epoch + 1}_decoder_model.pth"))

        gc.collect()

    # Plot training performance
    st.subheader("Training Performance")
    plt.plot(range(0, len(losses)), losses)
    plt.xlabel('[Training Epoch]')
    plt.xlim([0, len(losses)])
    plt.ylabel('[Reconstruction Error]')
    plt.title('AENN Training Performance')
    st.pyplot(plt)

    # Evaluation
    encoder_eval = Encoder()
    decoder_eval = Decoder()
    encoder_eval.load_state_dict(torch.load(os.path.join("models", "ep_1_encoder_model.pth")))
    decoder_eval.load_state_dict(torch.load(os.path.join("models", "ep_1_decoder_model.pth")))

    data = autograd.Variable(torch_dataset)
    encoder_eval.eval()
    decoder_eval.eval()
    reconstruction = decoder_eval(encoder_eval(data))
    reconstruction_loss_all = loss_function(reconstruction, data)
    st.write(f"[LOG {now}] Reconstruction loss: {reconstruction_loss_all.item()}")

    reconstruction_loss_transaction = np.zeros(reconstruction.size()[0])
    for i in range(reconstruction.size()[0]):
        reconstruction_loss_transaction[i] = loss_function(reconstruction[i], data[i]).item()

    # Plot results
    st.subheader("Anomaly Detection Results")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_data = np.column_stack((np.arange(len(reconstruction_loss_transaction)), reconstruction_loss_transaction))
    regular_data = plot_data[label == 'regular']
    global_outliers = plot_data[label == 'global']
    local_outliers = plot_data[label == 'local']
    ax.scatter(regular_data[:, 0], regular_data[:, 1], c='C0', alpha=0.4, marker="o", label='regular')
    ax.scatter(global_outliers[:, 0], global_outliers[:, 1], c='C1', marker="^", label='global')
    ax.scatter(local_outliers[:, 0], local_outliers[:, 1], c='C2', marker="s", label='local')
    ax.set_xlabel('[Transaction ID]')
    ax.set_ylabel('[Reconstruction error]')
    ax.set_title('AENN Transaction Anomaly Detection Results')
    ax.legend(loc="best")
    st.pyplot(fig)

    # 3D Scatter plot
    st.subheader("3D Scatter Plot of Encoded Data")
    encoded_data = encoder_eval(data).data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_data[label == 'regular', 0], encoded_data[label == 'regular', 1], encoded_data[label == 'regular', 2], c='C0', alpha=0.4, marker="o", label='regular')
    ax.scatter(encoded_data[label == 'global', 0], encoded_data[label == 'global', 1], encoded_data[label == 'global', 2], c='C1', marker="^", label='global')
    ax.scatter(encoded_data[label == 'local', 0], encoded_data[label == 'local', 1], encoded_data[label == 'local', 2], c='C2', marker="s", label='local')
    ax.set_xlabel('[Encoded dimension 1]')
    ax.set_ylabel('[Encoded dimension 2]')
    ax.set_zlabel('[Encoded dimension 3]')
    ax.set_title('AENN Encoded Data Visualization')
    st.pyplot(fig)
