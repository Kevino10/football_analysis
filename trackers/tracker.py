from ultralytics import YOLO
import supervision as sv
import json
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Stocke les couleurs des équipes
        self.player_team_dict = {}  # Stocke l'équipe assignée à chaque joueur
        self.kmeans = None  # Modèle de clustering
        self.referee_ids = set()  # Liste des arbitres sélectionnés
        self.next_id = 1  # ID auto-incrémenté pour les objets sans ID

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        top_half = player_crop[:player_crop.shape[0] // 2]
        return np.mean(top_half, axis=(0, 1))

    def manual_object_selection(self, frame, player_detections):
        """Permet de sélectionner manuellement les joueurs et les arbitres"""
        if not player_detections:
            print("⚠️ Aucun joueur détecté. Recherche d'une autre frame...")
            return False
        
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Cliquez sur un joueur de chaque équipe et un arbitre")
        coords = plt.ginput(3, timeout=30)
        plt.close()

        selected_colors = []
        selected_referees = []
        
        for idx, (x, y) in enumerate(coords):
            x, y = int(x), int(y)
            for player_id, data in player_detections.items():
                bbox = data["bbox"]
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    if idx < 2:  # Sélection des joueurs d'équipe
                        selected_colors.append(self.get_player_color(frame, bbox))
                    elif idx == 2:  # Sélection des arbitres
                        selected_referees.append(player_id)
                    break
        
        if len(selected_colors) == 2:
            self.team_colors = {1: selected_colors[0], 2: selected_colors[1]}
            self.referee_ids = set(selected_referees)
            print("✅ Équipes et arbitres assignés manuellement !")
            return True
        return False

    def get_player_team(self, frame, player_bbox, player_id):
        """Retourne l'équipe d'un joueur donné en fonction de sa couleur"""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        distances = [np.linalg.norm(player_color - color) for color in self.team_colors.values()]
        team_id = np.argmin(distances) + 1
        self.player_team_dict[player_id] = team_id
        return team_id

class Tracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tracker = sv.ByteTrack()
        self.tracking_data = []
        self.team_assigner = TeamAssigner()
        self.homography_matrix = np.load('calibration/matrices/matrice_match_du_jour.npy')
        print("✅ Matrice de calibration chargée avec succès.")


    def apply_homography(self, x, y):
        point = np.array([[x, y]], dtype="float32").reshape(1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return float(transformed_point[0][0][0]), float(transformed_point[0][0][1])


    def detect_frames(self, frames):
        model = YOLO(self.model_path)
        detections = [model.predict(frame, conf=0.3, verbose=True)[0] for frame in frames]
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        first_frame = None
        player_detections = {}
        
        for frame in frames:
            detection_result = detections[frames.index(frame)]
            if detection_result is not None and hasattr(detection_result, 'boxes'):
                detection_supervision = sv.Detections.from_ultralytics(detection_result)
                if detection_supervision is not None and len(detection_supervision) > 0:
                    first_frame = frame
                    print(f"🔍 Détection trouvée sur frame {frames.index(frame)}")
                    print(f"📌 Objets détectés : {[d for d in detection_supervision]}")
                    
                    for d in detection_supervision:
                        track_id = d[4] if d[4] is not None else self.team_assigner.next_id
                        self.team_assigner.next_id += 1
                        player_detections[track_id] = {"bbox": d[0]}
                    
                    break
        
        if first_frame is None or not self.team_assigner.manual_object_selection(first_frame, player_detections):
            print("❌ Impossible d'assigner les équipes ou arbitres.")
            return {"players": [], "ball": []}
        
        for frame_num, (detection, frame) in enumerate(zip(detections, frames)):
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            frame_data = {"frame": frame_num + 1, "players": [], "ball": None}
            
            for det in detection_with_tracks:
                bbox, _, _, cls_id, track_id, _ = det
                if track_id in self.team_assigner.referee_ids:
                    continue  # Exclure les arbitres du suivi des joueurs
                
                x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                x, y = self.apply_homography(x, y)
                
                if cls_names[cls_id] == "ball":
                    frame_data["ball"] = {"x": x, "y": y, "bbox": bbox.tolist()}
                else:
                    team_id = self.team_assigner.get_player_team(frame, bbox, track_id)
                    frame_data["players"].append({"id": int(track_id), "x": x, "y": y, "bbox": bbox.tolist(), "team": f"team_{team_id}"})
            
            self.tracking_data.append(frame_data)
        
        return {"players": [f["players"] for f in self.tracking_data], "ball": [f["ball"] if f["ball"] else {} for f in self.tracking_data]}

    def save_tracking_data_xml(self, path):
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom.minidom import parseString

            root = Element("match")
            for frame in self.tracking_data:
                frame_elem = SubElement(root, "frame", id=str(frame["frame"]))
                for player in frame["players"]:
                    player_elem = SubElement(frame_elem, "player", id=str(player["id"]), team=player["team"])
                    player_elem.set("x", str(player["x"]))
                    player_elem.set("y", str(player["y"]))

                if frame["ball"]:
                    ball_elem = SubElement(frame_elem, "ball")
                    ball_elem.set("x", str(frame["ball"]["x"]))
                    ball_elem.set("y", str(frame["ball"]["y"]))

            with open(path, 'w') as f:
                f.write(parseString(tostring(root)).toprettyxml())

            print(f"✅ Données enregistrées dans {path}")
