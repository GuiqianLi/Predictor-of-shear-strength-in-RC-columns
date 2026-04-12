#!/usr/bin/env python
# coding: utf-8

# # Prediction of shear strength in RC columns

# In[8]:


# define some necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor 
import joblib
# impoting the strength database
from openpyxl import load_workbook
dataset = pd.read_excel('钢筋混凝土柱抗剪强度数据集机器学习输入-20250515 - 8参数.xlsx', engine='openpyxl')
dataset.head()
# checking the dimension of the database
print(dataset.shape)
# define the inputs and the output
X = dataset.loc[:, dataset.columns != 'Vu']
y = dataset.loc[:, 'Vu']
# randomly spliting the database into training-testing sets as 75%-25% 
from sklearn.model_selection import train_test_split,KFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=240)
# Training Ph3-XGBR model
model = XGBRegressor (n_estimators=525,
                           max_depth=2,
                           learning_rate=0.201,
                           subsample=0.88,
                           colsample_bytree=0.79,
                           gamma=0.208,
                           random_state=700) 
model.fit(X_train, y_train)
# Save the model to a file
joblib.dump(model, 'Prediction of shear strength in RC columns.pkl')
print("Model training completed and saved！")


# In[10]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 设置页面配置
st.set_page_config(
    page_title="RC柱抗剪强度预测系统",
    page_icon="🏗️",
    layout="wide"
)

# 标题与说明
st.title("🏗️ 钢筋混凝土柱抗剪强度预测系统")
st.markdown("""
本应用基于 **XGBoost 回归模型**，根据柱的几何、材料及荷载参数，预测其抗剪强度 $V_u$（kN）。  
请输入以下 8 个特征值，点击预测按钮即可获得结果。
""")

# 侧边栏：模型信息
with st.sidebar:
    st.header("📌 模型信息")
    st.markdown("""
    - **算法**：XGBoost Regressor  
    - **输入特征**：8 个参数  
    - **输出目标**：抗剪强度 $V_u$ (kN)  
    - **训练数据**：钢筋混凝土柱试验数据集  
    """)
    st.divider()
    st.caption("模型文件：Prediction of shear strength in RC columns.pkl")

# 加载模型（使用缓存，只加载一次）
@st.cache_resource
def load_model():
    model = joblib.load('Prediction of shear strength in RC columns.pkl')
    return model

try:
    model = load_model()
    st.success("✅ 模型加载成功！", icon="🎉")
except Exception as e:
    st.error(f"❌ 模型加载失败：{e}\n请确保 `Prediction of shear strength in RC columns.pkl` 文件存在于当前目录。")
    st.stop()

# 创建输入表单（使用两列布局）
st.header("📊 输入参数")
col1, col2 = st.columns(2)

with col1:
    L = st.number_input(
        "**L** - 柱长度 / mm",
        min_value=0.0, max_value=5000.0, value=1000.0,
        help="柱的净高或长度，单位 mm"
    )
    fc = st.number_input(
        "**fc** - 混凝土抗压强度 / MPa",
        min_value=0.0, max_value=200.0, value=25.0,
        help="混凝土圆柱体抗压强度，单位 MPa"
    )
    ρs = st.number_input(
        "**ρs** - 体积配箍率",
        min_value=0.0, max_value=1.0, value=0.02, format="%.4f",
        help="箍筋体积与核心混凝土体积之比"
    )
    P = st.number_input(
        "**P** - 轴压力 / kN",
        min_value=0.0, max_value=10000.0, value=500.0,
        help="施加的轴向压力，正值表示压力，单位 kN"
    )

with col2:
    Vc = st.number_input(
        "**Vc** - 混凝土贡献的抗剪强度 / kN",
        min_value=0.0, max_value=5000.0, value=200.0,
        help="混凝土部分提供的抗剪承载力"
    )
    Vs = st.number_input(
        "**Vs** - 箍筋贡献的抗剪强度 / kN",
        min_value=0.0, max_value=5000.0, value=150.0,
        help="箍筋部分提供的抗剪承载力"
    )
    Vl = st.number_input(
        "**Vl** - 纵筋贡献的抗剪强度 / kN",
        min_value=0.0, max_value=5000.0, value=300.0,
        help="纵向钢筋部分提供的抗剪承载力"
    )
    Vp = st.number_input(
        "**Vp** - 轴压力贡献的抗剪强度 / kN",
        min_value=0.0, max_value=5000.0, value=0.0,
        help="轴压力（拱效应）贡献的抗剪承载力"
    )

# 将输入特征转换为 DataFrame（与训练格式一致）
input_data = pd.DataFrame({
    'L': [L],
    'fc': [fc],
    'ρs': [ρs],
    'P': [P],
    'Vc': [Vc],
    'Vs': [Vs],
    'Vl': [Vl],
    'Vp': [Vp]
})

# 预测按钮
st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)

# 执行预测并显示结果
if predict_btn:
    with st.spinner("模型推理中，请稍候..."):
        prediction = model.predict(input_data)[0]
    
    # 显示结果
    with col_res:
        st.success("### 预测结果")
        st.metric(
            label="📐 抗剪强度 $V_u$ (kN)",
            value=f"{prediction:.2f} kN",
            delta=None,
            delta_color="normal"
        )
        
        # 可选：给出强度等级建议（根据实际工程经验自定义）
        if prediction < 100:
            st.warning("⚠️ 预测强度较低，建议检查截面尺寸或配箍率")
        elif prediction > 800:
            st.info("📈 预测强度较高，抗剪能力充足")
        else:
            st.success("✅ 预测强度在常规范围内")

# 显示输入参数汇总（可选）
with st.expander("📋 查看输入参数汇总"):
    st.dataframe(input_data.style.highlight_max(axis=1))
    st.caption("注：以上数值均经过四舍五入，预测时使用原始输入值。")

# 页脚说明
st.divider()
st.caption("本模型仅用于科研参考，实际工程应用请结合规范校核。")


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Predictor of RC Column Shear Strength",
    page_icon="🏗️",
    layout="wide"
)

# Title and Description
st.title("🏗️ Predictor of RC Column Shear Strength")
st.markdown("""
This application is based on the Ph3-XGBR model to predict the shear strength $V_u$ (kN) of RC columns.
Please enter the following 8 feature values and click the predict button to obtain the result.
""")

# Model Information
with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
     - **Input**：8 features  
     - **Output**：Shear strength $V_u$ (kN)  
       """)
    st.divider()
    st.caption("Model file：Prediction of shear strength in RC columns.pkl")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('Prediction of shear strength in RC columns.pkl')
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully！", icon="🎉")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}\nPlease ensure the `Prediction of shear strength in RC columns.pkl`  file exists in the current directory.")
    st.stop()

# Create input form
st.header("📊 Input features")
col1, col2 = st.columns(2)

with col1:
    L = st.number_input(
        "**L** - column height / mm",
        min_value=80.0, max_value=1600.0, value=1000.0,
       
    )
    fc = st.number_input(
        "**fc** - concrete compressive strength / MPa",
        min_value=4.0, max_value=216.0, value=25.0,
      
    )
    ρs = st.number_input(
        "**ρs** - transverse reinforcement ratio(%)",
        min_value=0.0, max_value=2.0, value=0.02, format="%.4f",
      
    )
    P = st.number_input(
        "**P** - axial load / kN",
        min_value=0.0, max_value=18400.0, value=500.0,
       
    )

with col2:
    Vc = st.number_input(
        "**Vc** - shear contribution of concrete / kN",
        min_value=32.0, max_value=2830.0, value=200.0,
       
    )
    Vs = st.number_input(
        "**Vs** - shear contribution of transverse reinforcement / kN",
        min_value=0.0, max_value=3580.0, value=150.0,
       
    )
    Vl = st.number_input(
        "**Vl** - shear contribution of longitudinal reinforcement / kN",
        min_value=25.0, max_value=11400.0, value=300.0,
      
    )
    Vp = st.number_input(
        "**Vp** - shear contribution of axial force / kN",
        min_value=0.0, max_value=2520.0, value=0.0,
      
    )

# Convert input features into a DataFrame
input_data = pd.DataFrame({
    'L': [L],
    'fc': [fc],
    'ρs': [ρs],
    'P': [P],
    'Vc': [Vc],
    'Vs': [Vs],
    'Vl': [Vl],
    'Vp': [Vp]
})

# Prediction button
st.markdown("---")
col_btn, col_res = st.columns([1, 3])
with col_btn:
    predict_btn = st.button("🔮 Start predicting", type="primary", use_container_width=True)

# Execute predictions and display results
if predict_btn:
    with st.spinner("Prediction in progress, please wait..."):
        prediction = model.predict(input_data)[0]
   
    with col_res:
        st.success("### Prediction result")
        st.metric(
            label="📐 Shear strength $V_u$ (kN)",
            value=f"{prediction:.2f} kN",
            delta=None,
            delta_color="normal"
        )


# In[9]:


#用最佳参数进行预测,并打印训练集和测试集相关参数


Z1 = model.predict(X_train)
Z2 = model.predict(X_test)
Z3 = model.predict(X)

# 输出训练集指标
print(
    "Training R2:", r2_score(y_train, Z1),
    "MSE:", mean_squared_error(y_train, Z1),
    "RMSE:", np.sqrt(mean_squared_error(y_train, Z1)),
    "MAE:", mean_absolute_error(y_train, Z1),
    "MAPE:", MAPE(y_train, Z1)
)

# 输出测试集指标
print(
    "Testing R2:", r2_score(y_test, Z2),
    "MSE:", mean_squared_error(y_test, Z2),
    "RMSE:", np.sqrt(mean_squared_error(y_test, Z2)),
    "MAE:", mean_absolute_error(y_test, Z2),
    "MAPE:", MAPE(y_test, Z2)
)
# 输出总集指标
print(
    "Totalset R2:", r2_score(y, Z3),
    "MSE:", mean_squared_error(y, Z3),
    "RMSE:", np.sqrt(mean_squared_error(y, Z3)),
    "MAE:", mean_absolute_error(y, Z3),
    "MAPE:", MAPE(y, Z3)
)


# In[7]:


# define some necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor 
import joblib
# impoting the strength database
from openpyxl import load_workbook
dataset = pd.read_excel('钢筋混凝土柱抗剪强度数据集机器学习输入-20250515 - 8参数.xlsx', engine='openpyxl')
dataset.head()


# In[ ]:


# checking the dimension of the database
print(dataset.shape)
# define the inputs and the output
X = dataset.loc[:, dataset.columns != 'Vu']
y = dataset.loc[:, 'Vu']
# randomly spliting the database into training-testing sets as 75%-25% 
from sklearn.model_selection import train_test_split,KFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=240)
# Training Ph3-XGBR model
model = XGBRegressor (n_estimators=525,
                           max_depth=2,
                           learning_rate=0.201,
                           subsample=0.88,
                           colsample_bytree=0.79,
                           gamma=0.208,
                           random_state=700) 
model.fit(X_train, y_train)
# Save the model to a file
joblib.dump(model, 'Prediction of shear strength in RC columns.pkl')
print("Model training completed and saved！")

