import streamlit as st

st.title("🚀 My Portfolio Home")
cols = st.columns(4)
apps = [
    ("Business Opp.", "app.py"),
    ("Cannabis Rsch", "canna.py"),
    ("SVJ Analysis",  "s_v_j.py"),
    ("Synth Data",    "neymar.py"),
]
for col, (name, path) in zip(cols, apps):
    with col:
        st.image(f"thumbnails/{name}.png")
        if st.button(name):
            st.markdown(f'<iframe src="https://share.streamlit.io/your-username/your-repo/main/{path}" width="100%" height="800"></iframe>', unsafe_allow_html=True)