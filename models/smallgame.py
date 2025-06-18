import streamlit as st
import streamlit.components.v1 as components

st.markdown("## 🎮 Mini-Game: Recommend While You Wait")
st.markdown("Dodge the 🍿 distractions and collect 🎯 recommendation points!")

with open("models/runner_game.html", "r", encoding="utf-8") as f:
    html_string = f.read()

components.html(html_string, height=300)
