# ==========================================================
# app.py
# โปรแกรมจำแนกโรค COVID จากภาพ X-ray
# ==========================================================
# หมายเหตุสำคัญเกี่ยวกับโมเดลของคุณ (ตรวจสอบไฟล์ .pkcls จริงแล้ว):
#
# โมเดลนี้ "ไม่ใช่" โมเดลที่รับค่าตัวแปรแบบกรอกฟอร์ม (age, gender, ฯลฯ)
# แต่เป็นผลลัพธ์จากเวิร์กโฟลว์ Orange Data Mining ที่ใช้ widget
# "Image Embedding" แปลงภาพ X-ray เป็นเวกเตอร์ตัวเลข 2,048 ค่า
# (ชื่อคอลัมน์ n0 ... n2047 ในไฟล์โมเดล) ก่อนนำไปฝึกโมเดล
# (Neural Network / SVM / Tree) ให้จำแนกเป็น 3 คลาส:
#     covid, normal, pneumonia
#
# ดังนั้น "ตัวแปรต้น" ที่แท้จริงคือ "ภาพ X-ray" ไม่ใช่ตัวเลขที่กรอกเอง
# แอปนี้จึงรับภาพจากผู้ใช้ แล้วสกัด embedding ด้วยโมเดล Inception V3
# (ImageNet pretrained, pooling='avg' -> เวกเตอร์ 2048 มิติ) ก่อนส่งต่อ
# ให้โมเดล .pkcls ทำนาย
#
# *** สำคัญ: ถ้าตอนฝึกโมเดลใน Orange คุณเลือก Embedder ตัวอื่น
#     (เช่น SqueezeNet, VGG16, VGG19, Painters, DeepLoc, OpenFace)
#     ต้องเปลี่ยนฟังก์ชัน extract_embedding() ด้านล่างให้ตรงกัน
#     มิฉะนั้นตัวเลข embedding จะไม่ตรงกับตอนฝึก และผลทำนายจะผิดเพี้ยน ***
# ==========================================================

import streamlit as st
import joblib
import numpy as np
import glob
import os
from PIL import Image

# ----------------------------------------------------------
# 1) ตั้งค่าหน้าเว็บและหัวข้อแอป
# ----------------------------------------------------------
st.set_page_config(page_title="โปรแกรมจำแนกโรค COVID จากภาพ X-ray", layout="centered")

st.title("โปรเเกรมจำเเนกโรค covid จากภาพ x-rey")

st.markdown(
    """
    อัปโหลดภาพ X-ray ปอด แล้วกดปุ่ม **"ทำนายผล"**
    ระบบจะจำแนกภาพออกเป็น 3 กลุ่ม: **COVID / Normal / Pneumonia**
    """
)

# ----------------------------------------------------------
# 2) ส่วนเลือกโมเดล (.pkcls) — โหลดด้วย joblib
# ----------------------------------------------------------
st.sidebar.header("⚙️ การตั้งค่าโมเดล")

MODEL_DIR = "Model"  # นำไฟล์ .pkcls ไปวางไว้ในโฟลเดอร์นี้ (อยู่ข้าง app.py)

model_files = []
if os.path.isdir(MODEL_DIR):
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkcls"))

model_options = ["-- อัปโหลดไฟล์โมเดลเอง --"] + [os.path.basename(f) for f in model_files]
selected_option = st.sidebar.selectbox("เลือกไฟล์โมเดล (.pkcls)", options=model_options)

model_path = None
if selected_option == "-- อัปโหลดไฟล์โมเดลเอง --":
    uploaded_model = st.sidebar.file_uploader("อัปโหลดไฟล์โมเดล (.pkcls)", type=["pkcls"])
    if uploaded_model is not None:
        temp_path = "temp_uploaded_model.pkcls"
        with open(temp_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model_path = temp_path
else:
    model_path = os.path.join(MODEL_DIR, selected_option)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path):
    """โหลดโมเดล Orange (.pkcls) ด้วย joblib
    หมายเหตุ: ไฟล์นี้ต้องใช้ไลบรารี Orange3 + scikit-learn
    ในเวอร์ชันที่รองรับการ unpickle จึงจะโหลดสำเร็จ"""
    return joblib.load(path)


model = None
if model_path is not None and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.sidebar.success(f"โหลดโมเดลสำเร็จ: {os.path.basename(model_path)}")
        # แสดงชื่อคลาสที่โมเดลทำนายได้ (ดึงจาก domain ของ Orange)
        class_names = list(model.domain.class_var.values)
        st.sidebar.caption(f"คลาสที่ทำนายได้: {', '.join(class_names)}")
        st.sidebar.caption(f"จำนวนฟีเจอร์ (embedding) ที่โมเดลต้องการ: {len(model.domain.attributes)}")
    except Exception as e:
        st.sidebar.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.sidebar.info(
            "ตรวจสอบว่าได้ติดตั้ง Orange3 และ scikit-learn ตามที่ระบุใน requirements.txt แล้ว"
        )
else:
    st.sidebar.warning("กรุณาเลือกหรืออัปโหลดไฟล์โมเดล .pkcls ก่อนใช้งาน")


# ----------------------------------------------------------
# 3) โหลดโมเดลสกัด embedding (Inception V3) — cache ไว้ใช้ซ้ำ
# ----------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดลสกัดคุณลักษณะภาพ (Inception V3)...")
def load_embedder():
    """
    โหลดโมเดล Inception V3 (pretrained บน ImageNet) แบบไม่มีเลเยอร์สุดท้าย
    ใช้ pooling='avg' เพื่อให้ได้เวกเตอร์ผลลัพธ์ขนาด 2048
    *** ต้องตรงกับ Embedder ที่ใช้ตอนฝึกโมเดลใน Orange ***
    """
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    return base_model


def extract_embedding(pil_image, embedder):
    """
    แปลงภาพ PIL เป็นเวกเตอร์ embedding ขนาด 2048 มิติ
    ขั้นตอน: resize เป็น 299x299 -> แปลงเป็น array -> preprocess_input ของ Inception V3
    -> ส่งเข้าโมเดลเพื่อสกัดคุณลักษณะ (feature vector)
    """
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    img = pil_image.convert("RGB").resize((299, 299))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    embedding = embedder.predict(arr, verbose=0)
    return embedding.flatten()  # shape: (2048,)


# ----------------------------------------------------------
# 4) ส่วนอัปโหลดภาพ X-ray
# ----------------------------------------------------------
st.header("📤 อัปโหลดภาพ X-ray")
uploaded_image = st.file_uploader("เลือกไฟล์ภาพ X-ray (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="ภาพ X-ray ที่อัปโหลด", use_container_width=True)


# ----------------------------------------------------------
# 5) ปุ่มทำนายผล
# ----------------------------------------------------------
if st.button("ทำนายผล"):
    if model is None:
        st.error("กรุณาเลือก/อัปโหลดโมเดลก่อนทำการทำนาย")
    elif uploaded_image is None:
        st.error("กรุณาอัปโหลดภาพ X-ray ก่อนทำการทำนาย")
    else:
        try:
            with st.spinner("กำลังสกัดคุณลักษณะจากภาพและทำนายผล..."):
                embedder = load_embedder()
                embedding_vector = extract_embedding(image, embedder)

                # ตรวจสอบว่าจำนวนมิติของ embedding ตรงกับที่โมเดลต้องการหรือไม่
                n_expected = len(model.domain.attributes)
                if embedding_vector.shape[0] != n_expected:
                    st.error(
                        f"จำนวนมิติของ embedding ({embedding_vector.shape[0]}) "
                        f"ไม่ตรงกับที่โมเดลต้องการ ({n_expected}). "
                        f"กรุณาตรวจสอบว่า Embedder ที่ใช้ตอนฝึกโมเดลตรงกับที่ใช้ในแอปนี้หรือไม่ "
                        f"(ดูคำอธิบายท้ายโค้ด)"
                    )
                else:
                    X_input = embedding_vector.reshape(1, -1)

                    # โมเดล Orange สามารถเรียกใช้กับ numpy array ได้โดยตรง
                    # ผลลัพธ์คือ index ของคลาส (ตาม class_var.values)
                    pred_idx = model(X_input)
                    class_names = list(model.domain.class_var.values)
                    pred_label = class_names[int(pred_idx[0])]

                    # พยายามดึงความน่าจะเป็นของแต่ละคลาส (ถ้าโมเดลรองรับ)
                    probs_text = ""
                    try:
                        from Orange.classification import Model
                        _, probs = model(X_input, ret=Model.ValueProbs)
                        prob_dict = {name: f"{p*100:.2f}%" for name, p in zip(class_names, probs[0])}
                        probs_text = " | ".join([f"{k}: {v}" for k, v in prob_dict.items()])
                    except Exception:
                        pass

                    # แสดงผลลัพธ์
                    if pred_label.lower() == "covid":
                        st.error(f"⚠️ ผลการทำนาย: **COVID**")
                    elif pred_label.lower() == "pneumonia":
                        st.warning(f"🔶 ผลการทำนาย: **Pneumonia (ปอดอักเสบ)**")
                    else:
                        st.success(f"✅ ผลการทำนาย: **Normal (ปกติ)**")

                    if probs_text:
                        st.caption(f"ความน่าจะเป็นแต่ละคลาส: {probs_text}")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
