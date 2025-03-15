import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance
import supervision as sv
from calibration.calibration import compute_homography  # 🔹 Importer la fonction pour recalculer l'homographie dynamiquement

class CameraMovementEstimator():
    def __init__(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("❌ Erreur : Frame d'entrée invalide, vérifie le chargement des vidéos.")

        if len(frame.shape) < 3 or frame.shape[2] != 3:
            raise ValueError(f"❌ Erreur : L’image fournie ne contient pas 3 canaux (attendu: BGR). Shape: {frame.shape}")

        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔹 Correction : Ajout de `self.features`
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7
        )


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calcule le mouvement de la caméra entre les frames et stocke les décalages X, Y.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old in zip(new_features, old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    

    def update_homography_per_frame(self, frames):
        """
        Recalcule dynamiquement la matrice homographique en tenant compte du mouvement de la caméra.
        """
        homographies = []
        for frame_num, frame in enumerate(frames):
            homography_matrix, detected_corners = compute_homography(frame)
            if homography_matrix is not None:
                homographies.append(homography_matrix)
            else:
                if len(homographies) > 0:
                    homographies.append(homographies[-1])  # 🔹 Utiliser la dernière matrice connue si l'image est mauvaise
                else:
                    raise ValueError("⚠️ Impossible de calculer la première matrice d'homographie.")

        print(f"✅ Matrices homographiques recalculées pour {len(frames)} frames.")
        return homographies


    def apply_dynamic_homography(self, x, y, frame_num, homographies, camera_movement_per_frame):
        """
        Applique la transformation homographique mise à jour pour chaque frame, avec correction du mouvement de la caméra.
        """
        camera_movement = camera_movement_per_frame[frame_num]  # 🔹 Correction basée sur la caméra
        x_adjusted = x - camera_movement[0]
        y_adjusted = y - camera_movement[1]

        point = np.array([[x_adjusted, y_adjusted]], dtype="float32").reshape(1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, homographies[frame_num])
        
        return float(transformed_point[0][0][0]), float(transformed_point[0][0][1])


    def draw_camera_movement(self, output_video_frames, camera_movement_per_frame):
        """
        Dessine les mouvements de la caméra sur les frames vidéo.
        """
        for frame_num, frame in enumerate(output_video_frames):
            if frame_num < len(camera_movement_per_frame):
                movement = camera_movement_per_frame[frame_num]
                x_movement, y_movement = movement
                # Dessiner les mouvements sous forme de flèches
                height, width, _ = frame.shape
                start_point = (int(width / 2), int(height / 2))
                end_point = (int(width / 2 + x_movement), int(height / 2 + y_movement))
                color = (0, 0, 255)  # Couleur rouge pour la flèche
                thickness = 2
                cv2.arrowedLine(frame, start_point, end_point, color, thickness)
            else:
                print(f"Frame {frame_num} n'a pas de données de mouvement, pas de flèche dessinée.")
        return output_video_frames
