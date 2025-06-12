import streamlit as st
import time
import random

st.set_page_config(page_title="Doodle Jump", layout="wide")

# Constants
HEIGHT = 10
OBSTACLE_PROB = 0.2

# Initialize game state
if 'pos' not in st.session_state:
    st.session_state.pos = HEIGHT // 2
    st.session_state.jump = False
    st.session_state.obstacles = [False] * HEIGHT
    st.session_state.score = 0
    st.session_state.game_over = False

# Obstacle logic
def spawn_obstacle():
    return random.random() < OBSTACLE_PROB

# Display game frame
def render_game():
    frame = ["⬛"] * HEIGHT
    if st.session_state.obstacles[st.session_state.pos]:
        frame[st.session_state.pos] = "💥"  # Collision
        st.session_state.game_over = True
    else:
        frame[st.session_state.pos] = "🙂"  # Player

    for i in range(HEIGHT):
        if st.session_state.obstacles[i] and i != st.session_state.pos:
            frame[i] = "🚧"

    st.write("### Game Area")
    for line in reversed(frame):
        st.text(line)

# Game logic
def game_loop():
    if st.session_state.jump:
        st.session_state.pos = min(HEIGHT - 1, st.session_state.pos + 1)
        st.session_state.jump = False
    else:
        st.session_state.pos = max(0, st.session_state.pos - 1)

    # Scroll obstacles
    st.session_state.obstacles.pop(0)
    st.session_state.obstacles.append(spawn_obstacle())
    st.session_state.score += 1

    render_game()
    st.write(f"**Score:** {st.session_state.score}")

# Jump button
col1, col2 = st.columns([2, 1])
with col1:
    placeholder = st.empty()
    for _ in range(100):  # max game ticks
        if st.session_state.game_over:
            break
        with placeholder.container():
            game_loop()
        time.sleep(0.5)
        placeholder.empty()

with col2:
    if st.button("⬆️ Jump"):
        st.session_state.jump = True

if st.session_state.game_over:
    st.error("💀 Game Over!")
    if st.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()
