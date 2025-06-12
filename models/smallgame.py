import streamlit as st
import streamlit.components.v1 as components

with open("runner_game.html", 'r') as f:
    html_string = f.read()

components.html(html_string, height=250)