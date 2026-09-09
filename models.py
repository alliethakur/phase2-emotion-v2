# models.py - Model architecture and loading
import streamlit as st
import torch
import torch.nn as nn
import timm

class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    def forward(self, x):
        return self.base(x)

class AudioEmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    face_model = FaceEmotionModel()
    face_model.load_state_dict(torch.load('face_model.pth', map_location=device))
    face_model.eval()
    audio_model = AudioEmotionLSTM()
    audio_model.load_state_dict(torch.load('audio_model.pth', map_location=device))
    audio_model.eval()
    return face_model, audio_model