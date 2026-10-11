import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Tambaqui Analyzer",
    page_icon="🐟"
)

st.title("🐟 Tambaqui Analyzer")

video = st.file_uploader(
    "Selecione um vídeo",
    type=["mp4","avi","mov"]
)

if video:

    st.video(video)

    if st.button("Analisar"):

        dados = pd.DataFrame(
            {
                "Frame":[1,2,3],
                "ID":[1,2,3],
                "X":[120,125,130],
                "Y":[210,215,218]
            }
        )

        st.success(
            "Vídeo processado!"
        )

        st.dataframe(dados)

        st.download_button(
            "Download CSV",
            dados.to_csv(index=False),
            "dados.csv"
        )
