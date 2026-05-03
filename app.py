import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from your_model_script import get_model # 导入你第三阶段定义的模型结构

# 1. 页面配置
st.set_page_config(page_title="糖尿病视网膜病变诊断系统", layout="centered")
st.title("👁️ DR 智能分级诊断系统")
st.write("上传眼底扫描图，AI 将自动分析病变程度。")

# 2. 加载模型权重
@st.cache_resource
def load_ai_model():
    # 确保这里的 num_classes 和你生成权重时一致
    model = get_model(num_classes=5) 
    
    # 加载权重
    state_dict = torch.load("./best_model.pth", map_location=torch.device('cpu'))
    
    # 这一步是关键：强制匹配
    model.load_state_dict(state_dict, strict=False) 
    
    model.eval()
    return model

model = load_ai_model()

# 3. 定义与第二阶段一致的预处理逻辑
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. 侧边栏及文件上传
uploaded_file = st.sidebar.file_uploader("选择眼底图像...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 展示上传的图片
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上传的原始图像', use_column_width=True)
    
    # 按钮：开始诊断
    if st.button('开始 AI 自动诊断'):
        with st.spinner('正在分析图像特征...'):
            # 预处理并预测
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                confidence, prediction = torch.max(probabilities, 0)
            
            # 5. 结果展示
            st.divider()
            classes = ['0 - 正常 (Normal)', '1 - 轻度 (Mild)', '2 - 中度 (Moderate)', '3 - 重度 (Severe)', '4 - 增殖期 (PDR)']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测分级", classes[prediction.item()])
            with col2:
                st.metric("置信度", f"{confidence.item()*100:.2f}%")
            
            # 概率分布图
            st.write("### 各等级概率分布：")
            st.bar_chart(probabilities.numpy())
            
            if prediction.item() >= 3:
                st.error("⚠️ 检测到严重病变风险，建议立即咨询眼科专家。")
            else:
                st.success("✅ 诊断完成。")
