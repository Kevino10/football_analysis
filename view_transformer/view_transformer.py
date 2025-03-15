import numpy as np
import cv2
import supervision as sv

class ViewTransformer():
    def __init__(self):
        court_width = 68  # Largeur du terrain
        court_length = 23.32  # Longueur du terrain

        # Coordonnées des 4 coins du terrain dans l'image (pixel)
        self.pixel_vertices = np.array([[110, 1035], 
                                        [265, 275], 
                                        [910, 260], 
                                        [1640, 915]])

        # Coordonnées des 4 coins du terrain dans la perspective souhaitée
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Conversion en float32 pour la transformation
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calcul de la matrice de transformation perspective
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        """Transforme un point à partir des coordonnées de l'image à celles du terrain"""
        p = (int(point[0]), int(point[1]))  # Conversion en entiers
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0  # Vérifie si le point est à l'intérieur
        if not is_inside:
            return None  # Si le point est en dehors du terrain

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)  # Reshape du point pour la transformation
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)  # Retourne le point transformé

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if isinstance(track, sv.Detections):  # Assurez-vous qu'il s'agit d'une détection
                    for track_info in track:
                        print(f"Track info for track_id {track_info}: {type(track_info)}")  # Affiche le type et le contenu

                        # Vérifier si track_info est un dictionnaire avant d'y accéder
                        if isinstance(track_info, dict):
                            if 'position_adjusted' in track_info:
                                position = track_info['position_adjusted']
                            else:
                                print(f"Position ajustée manquante pour track_id {track_info.get('tracker_id', 'Unknown')}, utiliser des valeurs par défaut.")
                                position = [0, 0, 0, 0]  # Valeurs par défaut si la position ajustée est manquante
                        else:
                            print(f"Track info n'est pas un dictionnaire pour le track_id, type trouvé: {type(track_info)}")
                            position = [0, 0, 0, 0]  # Valeurs par défaut si ce n'est pas un dictionnaire

                        # Appliquez ici la transformation de la position si nécessaire
                        # transformed_position = self.transform_point(position) 
                        # track_info['transformed_position'] = transformed_position
                else:
                    print(f"Le track {frame_num} n'est pas un objet Detections, type trouvé: {type(track)}")
