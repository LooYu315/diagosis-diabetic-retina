import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

# --- 1. 还原模型结构 ---
def get_model(num_classes=5):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

@st.cache_resource
def load_ensemble():
    ensemble = []
    # 注意：确保你的.pth文件放在仓库的 models 文件夹下，或者修改下方路径
    for i in range(5):
        path = f"best_model_fold{i}.pth" # 如果你放在根目录就这么写
        m = get_model(5)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        ensemble.append(m)
    return ensemble

# --- 2. 还原 Ben Graham + CLAHE 预处理 ---
def preprocess_for_ai(image):
    # PIL 转 CV2
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 裁剪黑边
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    if np.any(mask):
        img = img[np.ix_(mask.any(1), mask.any(0))]
    
    # Resize & Ben Graham 增强
    img = cv2.resize(img, (512, 512))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    
    # CLAHE 增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.fromarray(img)).unsqueeze(0)

# --- 3. Streamlit 界面 ---
st.set_page_config(page_title="DR 集成诊断系统")
st.title("👁️ 糖尿病视网膜病变智能诊断")

uploaded_file = st.file_uploader("上传眼底照片", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="待诊断图像", use_column_width=True)
    
    if st.button("开始联合会诊预测"):
        with st.spinner("5个 DenseNet 模型正在计算平均概率..."):
            # 执行预处理
            input_tensor = preprocess_for_ai(image)
            models_list = load_ensemble()
            
            # 运行集成推理
            all_probs = []
            with torch.no_grad():
                for m in models_list:
                    output = m(input_tensor)
                    all_probs.append(torch.softmax(output, dim=1))
                
                # 平均概率
                avg_probs = torch.mean(torch.stack(all_probs), dim=0)
                prediction = torch.argmax(avg_probs, dim=1).item()
                confidence = torch.max(avg_probs).item()
            
            # 结果展示
            classes = ['0 - 正常', '1 - 轻度', '2 - 中度', '3 - 重度', '4 - 增殖期']
            st.metric("诊断结论", classes[prediction])
            st.progress(confidence)
            st.write(f"预测置信度: {confidence*100:.2f}%")
