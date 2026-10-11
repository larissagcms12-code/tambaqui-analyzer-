import streamlit as st
import pandas as pd
import cv2
import tempfile
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Tambaqui Behavior Analyzer",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Tambaqui Behavior Analyzer")

st.sidebar.header("Configurações")

conf_threshold = st.sidebar.slider(
    "Confiança YOLO",
    0.1,
    1.0,
    0.25
)

st.sidebar.markdown("---")

uploaded_video = st.file_uploader(
    "Upload do vídeo",
    type=["mp4", "avi", "mov"]
)

zona_x = st.sidebar.number_input(
    "Zona Alimentação X",
    value=100
)

zona_y = st.sidebar.number_input(
    "Zona Alimentação Y",
    value=100
)

zona_w = st.sidebar.number_input(
    "Largura",
    value=250
)

zona_h = st.sidebar.number_input(
    "Altura",
    value=250
)

if uploaded_video:

    st.video(uploaded_video)

    if st.button("▶ Iniciar Análise"):

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(
            uploaded_video.read()
        )

        temp_video.close()

        model = YOLO("modelos/best.pt")

        progress = st.progress(0)

        resultados = []

        cap = cv2.VideoCapture(
            temp_video.name
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_num = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_num += 1

            output = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_threshold
            )

            boxes = output[0].boxes

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()

                for fish_id, coord in zip(
                    ids,
                    coords
                ):

                    x1,y1,x2,y2 = coord

                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)

                    dentro = (
                        zona_x <= cx <= zona_x+zona_w
                        and
                        zona_y <= cy <= zona_y+zona_h
                    )

                    resultados.append([
                        frame_num,
                        int(fish_id),
                        cx,
                        cy,
                        dentro
                    ])

            progresso = int(
                frame_num / total_frames * 100
            )

            progress.progress(
                min(progresso,100)
            )

        cap.release()

        df = pd.DataFrame(
            resultados,
            columns=[
                "Frame",
                "ID",
                "X",
                "Y",
                "Area_Alimentacao"
            ]
        )

        st.success(
            "Análise concluída!"
        )

        st.subheader(
            "Tabela de Dados"
        )

        st.dataframe(df.head(50))

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download CSV",
            csv,
            file_name="comportamento.csv"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Peixes Rastreados"
            )

            st.metric(
                "Indivíduos",
                df["ID"].nunique()
            )

        with col2:

            st.subheader(
                "Registros"
            )

            st.metric(
                "Detecções",
                len(df)
            )

        st.subheader(
            "Tempo na Área de Alimentação"
        )

        tempo = (
            df.groupby("ID")
            ["Area_Alimentacao"]
            .sum()
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        tempo.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel(
            "Frames"
        )

        st.pyplot(fig)

        st.subheader(
            "Mapa de Calor"
        )

        fig2, ax2 = plt.subplots(
            figsize=(8,6)
        )

        sns.kdeplot(
            x=df["X"],
            y=df["Y"],
            fill=True,
            cmap="viridis",
            ax=ax2
        )

        ax2.invert_yaxis()

        st.pyplot(fig2)

        os.unlink(
            temp_video.name
        )
