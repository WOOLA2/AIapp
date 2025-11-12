import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.set_title("Student Dropout Prediction", fontsize=16)
ax.axis('off')

def animate(frame):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    if frame < 50:  # 0-5s: Title
        ax.text(5, 50, "Student Dropout Predictor", ha='center', fontsize=20, fontweight='bold')
    elif frame < 150:  # 5-15s: Data Entry
        ax.text(5, 80, "Enter Data:", ha='center', fontsize=14)
        data = ["Age: 18", "GPA: 3.2", "Attendance: 85%"]
        for i, d in enumerate(data[:int((frame-50)/33)+1]):
            ax.text(5, 60 - i*15, d, ha='center')
    elif frame < 250:  # 15-25s: Prediction
        ax.text(5, 80, "Prediction:", ha='center', fontsize=14)
        probs = [0, 0, 0]
        step = (frame - 150) / 100
        probs[0] = min(70 * step, 70)  # Enrolled
        probs[1] = min(20 * step, 20)  # Graduate
        probs[2] = min(10 * step, 10)  # Dropout
        bars = ax.bar(['Enrolled', 'Graduate', 'Dropout'], probs, color=['green', 'gold', 'red'])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha='center')
    else:  # 25-30s: CTA
        ax.text(5, 50, "Prevent Dropouts!\nTry the Model Now", ha='center', fontsize=16)

ani = animation.FuncAnimation(fig, animate, frames=300, interval=100)  # 30s at 10fps
ani.save('prediction_video.mp4', writer='ffmpeg', fps=10)  # Need ffmpeg installed
print("Video saved as prediction_video.mp4")