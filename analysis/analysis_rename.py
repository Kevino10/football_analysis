import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
import os
import cv2

# Charger la matrice homographique
homography_matrix = np.load('calibration/matrices/matrice_match_du_jour.npy')

# Créer le dossier d'output
os.makedirs('analysis/outputs', exist_ok=True)

# Détection de la zone visible réelle
video_path = "input_videos/test_4_coins.mov"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise Exception(f"❌ Impossible de lire la vidéo : {video_path}")

original_points = np.array([
    [0, 0],
    [frame.shape[1], 0],
    [0, frame.shape[0]],
    [frame.shape[1], frame.shape[0]]
], dtype="float32")

transformed_points = cv2.perspectiveTransform(original_points.reshape(-1, 1, 2), homography_matrix).reshape(-1, 2)

# ⚠️ Normalisation des dimensions visibles pour correspondre au terrain réel
scaling_factor_x = 105 / (max(transformed_points[:, 0]) - min(transformed_points[:, 0]))
scaling_factor_y = 68 / (max(transformed_points[:, 1]) - min(transformed_points[:, 1]))

visible_width = (max(transformed_points[:, 0]) - min(transformed_points[:, 0])) * scaling_factor_x
visible_height = (max(transformed_points[:, 1]) - min(transformed_points[:, 1])) * scaling_factor_y

print(f"✅ Zone visible recalibrée : {visible_width:.2f}m x {visible_height:.2f}m")

# Charger le fichier de tracking XML
tree = ET.parse('output/tracking_data.xml')
root = tree.getroot()

# Analyse de la possession
total_frames = 0
possession_time = {"team_1": 0, "team_2": 0}

def get_team(player_id):
    for frame in root.findall('frame'):
        for player in frame.findall('player'):
            if int(player.get('id')) == player_id:
                return player.get('team')
    return None

for frame in root.findall('frame'):
    total_frames += 1
    ball = frame.find('ball')
    if ball is None:
        continue
    ball_x, ball_y = float(ball.get('x')) * scaling_factor_x, float(ball.get('y')) * scaling_factor_y

    closest_player = None
    min_distance = float('inf')

    for player in frame.findall('player'):
        player_x = float(player.get('x')) * scaling_factor_x
        player_y = float(player.get('y')) * scaling_factor_y
        distance = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_player = int(player.get('id'))

    if closest_player is not None:
        team = get_team(closest_player)
        if team:
            possession_time[team] += 1

# Convertir en secondes (1 frame = 1/30 s, ajuster selon FPS)
fps = 30
possession_time_seconds = {team: frames / fps for team, frames in possession_time.items()}

# Calcul du pourcentage de possession
total_possession = sum(possession_time_seconds.values())
possession_percentage = {team: (time / total_possession * 100) if total_possession > 0 else 0 for team, time in possession_time_seconds.items()}

print(f"⏳ Temps de possession:")
print(f"⚽ Team 1: {possession_time_seconds['team_1']:.2f} sec ({possession_percentage['team_1']:.2f}%)")
print(f"⚽ Team 2: {possession_time_seconds['team_2']:.2f} sec ({possession_percentage['team_2']:.2f}%)")

# Analyse trajectoire et distance parcourue par un joueur
player_id_target = 9
distance_parcourue = 0
player_positions = {"x": [], "y": []}

for frame in root.findall('frame'):
    for player in frame.findall('player'):
        if int(player.get('id')) == player_id_target:
            x = float(player.get('x')) * scaling_factor_x
            y = float(player.get('y')) * scaling_factor_y
            player_positions["x"].append(x)
            player_positions["y"].append(y)

# Calcul de la distance totale
for i in range(1, len(player_positions["x"])):
    dx = player_positions["x"][i] - player_positions["x"][i - 1]
    dy = player_positions["y"][i] - player_positions["y"][i - 1]
    distance_parcourue += np.sqrt(dx ** 2 + dy ** 2)

print(f"📏 Distance parcourue par le joueur {player_id_target}: {distance_parcourue:.2f} m")

# Création du Pitch
pitch = Pitch(pitch_type='custom', pitch_length=visible_width, pitch_width=visible_height, line_color='black', pitch_color='white')
fig, ax = pitch.draw(figsize=(10, 7))

# Tracer la trajectoire du joueur cible
ax.plot(player_positions["x"], player_positions["y"], color='red', linewidth=2, marker='o', markersize=4, label=f"Joueur {player_id_target}")
plt.legend(loc='upper right', fontsize=8)
plt.title(f'Trajectoire et Distance Parcourue - Joueur {player_id_target}')
plt.savefig(f'analysis/outputs/trajectoire_joueur_{player_id_target}.png')
plt.close()

print(f"✅ Trajectoire et distance du joueur {player_id_target} sauvegardées dans analysis/outputs/trajectoire_joueur_{player_id_target}.png")