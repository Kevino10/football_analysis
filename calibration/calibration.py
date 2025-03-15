import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dossier et fichier de la matrice
output_dir = "calibration/matrices"
os.makedirs(output_dir, exist_ok=True)
output_matrix_file = os.path.join(output_dir, "matrice_match_du_jour.npy")

# Chemin de la vidéo
video_path = "input_videos/08fd33_4.mp4"

# Points cliqués
selected_points = []
frame_index = 0
click_confirmed = False  # Variable pour éviter les doubles clics

# 📌 **Points de référence essentiels pour une meilleure précision**
reference_points = {
    "coin_haut_gauche": [0, 68],  # Coin haut gauche du terrain
    "coin_haut_droit": [105, 68],  # Coin haut droit du terrain
    "centre_terrain": [52.5, 34],  # Point central du terrain
    "rond_central_gauche": [43.5, 34],  # Bord gauche du rond central
    "rond_central_droit": [61.5, 34],  # Bord droit du rond central
    "ligne_mediane_bas": [52.5, 68],  # Bas de la ligne médiane
    "ligne_mediane_haut": [52.5, 0],  # Haut de la ligne médiane
    "surface_repar_penalty_gauche": [16.5, 34],  # Point du centre surface gauche
    "surface_repar_penalty_droite": [88.5, 34]  # Point du centre surface droite
}

# Fonction de sélection des points

def select_point(event, x, y, flags, param):
    global selected_points, frame_index, click_confirmed
    if event == cv2.EVENT_LBUTTONDOWN:
        if not click_confirmed:
            print(f"🔍 Point proposé : {x}, {y} sur frame {frame_index}")
            confirm = input("Confirmez ce point ? (o/n/s) : ")
            if confirm.lower() == 'o':
                selected_points.append((x, y, frame_index))
                click_confirmed = True  # Bloque le double enregistrement du clic
                print(f"✅ Point enregistré : {x}, {y} sur frame {frame_index}")
                frame_index = 0  # Remettre la vidéo au début après chaque sélection
                if len(selected_points) >= len(reference_points):
                    cv2.destroyAllWindows()
            elif confirm.lower() == 's':
                print("⏩ Point ignoré.")
                selected_points.append(None)  # Enregistre un saut sans supprimer les précédents
            elif confirm.lower() == 'n':
                print("❌ Point refusé, sélectionnez un autre.")
    elif event == cv2.EVENT_LBUTTONUP:
        click_confirmed = False  # Réinitialise après le relâchement

print("🔍 Sélectionnez les points essentiels (Appuyez sur 's' pour sauter un point si non visible) :")
for key in reference_points.keys():
    print(f"🔹 {key.replace('_', ' ').capitalize()}")

while len(selected_points) < len(reference_points):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception(f"❌ Impossible de lire la frame {frame_index}")
    
    cv2.imshow("Sélection des points", frame)
    cv2.setMouseCallback("Sélection des points", select_point)
    key = cv2.waitKey(0)
    if key == ord('d'):
        frame_index += 1
    elif key == ord('a'):
        frame_index = max(0, frame_index - 1)  # Empêcher de descendre en dessous de 0
cv2.destroyAllWindows()

# 🎮 **Fonction pour gérer la navigation et sélection des points**
def navigate_and_select():
    global frame_index, paused

    cap = cv2.VideoCapture(video_path)
    while any(v is None for v in selected_points.values()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise Exception(f"❌ Impossible de lire la frame {frame_index}")
        
        point_name = next(key for key, value in selected_points.items() if value is None)
        print(f"\n🎯 Capture du point : {point_name.replace('_', ' ').capitalize()}")

        cv2.imshow("Sélection des points", frame)
        cv2.setMouseCallback("Sélection des points", select_point, param=point_name)

        while True:
            key = cv2.waitKey(0)
            if key == ord('d'):  # Avancer d'une frame
                frame_index += 1
                break
            elif key == ord('a'):  # Reculer d'une frame
                frame_index = max(0, frame_index - 1)
                break
            elif key == ord('w'):  # Zoom (avancer de 10 frames)
                frame_index = min(frame_index + 10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                break
            elif key == ord('s'):  # Dézoom (reculer de 10 frames)
                frame_index = max(0, frame_index - 10)
                break
            elif key == ord(' '):  # Pause / Reprendre
                paused = not paused
                if paused:
                    print("⏸️ Pause activée. Appuyez sur espace pour continuer.")
                break
            elif key == ord('q'):  # Quitter
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

# Suppression des points ignorés
points_video = np.array([p[:2] for p in selected_points if p is not None], dtype="float32")
points_reference = np.array(list(reference_points.values())[:len(points_video)], dtype="float32")

# **Calcul de la matrice homographique**
homography_matrix, _ = cv2.findHomography(points_video, points_reference)

# **Transformation des points pour affichage correct**
transformed_points = cv2.perspectiveTransform(points_video.reshape(-1, 1, 2), homography_matrix).reshape(-1, 2)

# **Création de l'image du terrain avec les points projetés**
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 105)
ax.set_ylim(0, 68)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)
ax.plot([0, 105], [0, 0], color="black", linewidth=2)
ax.plot([0, 105], [68, 68], color="black", linewidth=2)
ax.plot([0, 0], [0, 68], color="black", linewidth=2)
ax.plot([105, 105], [0, 68], color="black", linewidth=2)
ax.plot([52.5, 52.5], [0, 68], color="black", linestyle="dashed", linewidth=2)
ax.add_patch(plt.Circle((52.5, 34), 9.15, color="black", fill=False, linewidth=2))
ax.add_patch(patches.Rectangle((0, 21.1), 16.5, 25.8, edgecolor="black", fill=False, linewidth=2))
ax.add_patch(patches.Rectangle((88.5, 21.1), 16.5, 25.8, edgecolor="black", fill=False, linewidth=2))

# Ajouter les points transformés avec correction des indices
for i, (x, y) in enumerate(transformed_points, start=1):
    ax.scatter(x, 68 - y, color='red', s=100, edgecolor='black', zorder=3)  # Correction de l'inversion verticale
    ax.text(x, 68 - y + 2, f"P{i}", ha='center', fontsize=8, color='black')

plt.title("📌 Vérification des points projetés sur le terrain")
plt.savefig("calibration/verification_projection.png")
plt.show()

# **Sauvegarde de la matrice**
np.save(output_matrix_file, homography_matrix)
print(f"✅ Matrice sauvegardée dans {output_matrix_file}")