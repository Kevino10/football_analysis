from utils import read_video, save_video
from trackers.tracker import Tracker
from visualizer import draw_annotations
import os
import numpy as np

def main():
    # 1. Charger la vidéo
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # 2. Initialisation du tracker
    tracker = Tracker('models/best.pt')  # Correction: Suppression de sample_frame

    # 3. Tracking et export JSON/XML
    tracks = tracker.get_object_tracks(video_frames)

    # Créer le dossier output s'il n'existe pas
    os.makedirs("output", exist_ok=True)

    tracker.save_tracking_data_xml("output/tracking_data.xml")

    # 4. Sécurisation : s'assurer que le tracking couvre toute la vidéo (frames manquantes = vides)
    total_frames = len(video_frames)
    
    # Vérification que toutes les clés existent bien
    if 'players' not in tracks:
        tracks['players'] = []
    if 'referees' not in tracks:
        tracks['referees'] = []
    if 'ball' not in tracks:
        tracks['ball'] = []

    while len(tracks['players']) < total_frames:
        tracks['players'].append([])
        tracks['referees'].append([])  # Ajouté pour éviter KeyError
        tracks['ball'].append({})

    print(f"📊 Nombre total de frames vidéo : {total_frames}")
    print(f"📊 Nombre total de frames trackées : {len(tracks['players'])}")

    for frame_num, players in enumerate(tracks['players']):
        print(f"🎥 Frame {frame_num} - Nombre de joueurs détectés : {len(players)}")

        if len(players) > 0:
            print(f"  ➡️ Exemple joueur : {players[0]}")

    # 5. Simulation d'une possession (temporaire ou à connecter avec ton système de possession réel)
    team_ball_control = np.zeros(total_frames)

    # 6. Dessin des annotations
    annotated_frames = draw_annotations(video_frames, tracks, team_ball_control)

    # 7. Sauvegarde de la vidéo annotée
    os.makedirs("output_videos", exist_ok=True)
    save_video(annotated_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
