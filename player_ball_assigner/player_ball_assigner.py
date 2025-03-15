import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
import numpy as np

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.previous_ball_team = None  # Garder en mémoire l'équipe en possession du ballon
    
    def assign_ball_to_player(self, players, ball_bbox):
        """Assigner la balle à un joueur basé sur la distance"""
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999  # Initialiser à une grande valeur pour trouver la distance minimale
        assigned_player = -1

        # Vérification des joueurs les plus proches du ballon
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Calcul de la distance entre le ballon et le joueur
            distance_left = measure_distance((player_bbox[0], player_bbox[1]), ball_position)  # Coin gauche du joueur
            distance_right = measure_distance((player_bbox[2], player_bbox[1]), ball_position)  # Coin droit du joueur
            distance = min(distance_left, distance_right)

            # Si la distance est inférieure à une valeur seuil, c'est ce joueur qui reçoit la balle
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    def update_ball_possession(self, tracks, frame_num, player_assigner):
        """
        Met à jour la possession de la balle pour chaque joueur.
        """
        # Vérifier si la frame pour le ballon existe
        if frame_num not in tracks['ball'] or len(tracks['ball'][frame_num]) < 2:
            print(f"Aucune donnée pour le ballon à la frame {frame_num}, appliquer des valeurs par défaut.")
            # Appliquer des valeurs par défaut
            return -1  # Si aucune donnée n'est présente pour cette frame, renvoyer une valeur par défaut
        
        # Si les données du ballon existent, récupérer les coordonnées
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Assurez-vous que l'index [1] existe pour la frame

        # Calculer la position du ballon
        ball_position = get_center_of_bbox(ball_bbox)

        # Le reste du traitement pour attribuer la possession de la balle
        assigned_player = player_assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)
        
        return assigned_player

