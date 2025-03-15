import cv2
import sys
sys.path.append('../')
import supervision as sv
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24  # Frame rate en FPS

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        last_positions = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames):
                if not isinstance(object_tracks[frame_num], sv.Detections):
                    print(f"Frame {frame_num} is not of type Detections, skipping.")
                    continue

                detections = object_tracks[frame_num]

                for track_id, detection in enumerate(detections):
                    if not hasattr(detection, 'boxes'):
                        print(f"No 'boxes' attribute found for track_id {track_id}. Skipping.")
                        continue
                    
                    boxes = detection.boxes.xyxy  # Coordonnées des boîtes
                    bbox = boxes[track_id] if track_id < len(boxes) else None
                    if bbox is None:
                        continue

                    # Calcul de la position du joueur
                    position = get_foot_position(bbox)

                    if track_id not in last_positions:
                        last_positions[track_id] = position
                        continue

                    # Calcul de la distance parcourue
                    distance = measure_distance(last_positions[track_id], position)
                    total_distance[track_id] = total_distance.get(track_id, 0) + distance

                    # Calcul de la vitesse (en m/s) et conversion en km/h
                    speed = distance * self.frame_rate  # distance / time => m/s, ensuite *3600 pour obtenir km/h
                    speed_kmh = speed * 3.6

                    # Ajouter la vitesse et la distance dans les informations du joueur
                    if track_id not in tracks["players"][frame_num]:
                        tracks["players"][frame_num][track_id] = {}
                    
                    tracks["players"][frame_num][track_id]['speed'] = speed_kmh
                    tracks["players"][frame_num][track_id]['distance'] = total_distance[track_id]

                    # Mettre à jour la position
                    last_positions[track_id] = position

                # Debugging: Imprimer la distance et la vitesse pour chaque joueur
                print(f"Frame {frame_num} - Track ID {track_id}: Speed = {speed_kmh:.2f} km/h, Distance = {total_distance[track_id]:.2f} m")

    def draw_speed_and_distance(self, output_video_frames, tracks):
        """
        Dessine la vitesse et la distance sous les joueurs sur chaque frame.
        Vérifie d'abord l'affichage du texte simple sous les joueurs pour le débogage.
        """
        for frame_num, frame in enumerate(output_video_frames):
            if frame_num < len(tracks["players"]):
                for track_id, track_info in tracks["players"][frame_num].items():
                    if 'speed' in track_info and 'distance' in track_info:
                        speed_text = f"{track_info['speed']:.2f} km/h"
                        distance_text = f"{track_info['distance']:.2f} m"
                        x, y = get_foot_position(track_info['bbox'])  # Position du joueur
                        print(f"Track ID {track_id}: Speed = {track_info['speed']:.2f} km/h, Distance = {track_info['distance']:.2f} m")
                        print(f"Position: ({x}, {y})")  # Pour déboguer et vérifier si la position est correcte

                        # Vérification si les coordonnées sont dans un bon intervalle pour l'affichage
                        if x > 0 and y > 0:  # Vérifier que les coordonnées sont dans la vidéo
                            cv2.putText(frame, speed_text, (int(x), int(y) + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, distance_text, (int(x), int(y) + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Retourner les frames modifiées
            return output_video_frames
