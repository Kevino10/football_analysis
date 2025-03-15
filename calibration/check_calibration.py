import cv2
import numpy as np

# Charger la matrice homographique
homography_matrix = np.load('calibration/matrices/matrice_match_du_jour.npy')
print("✅ Matrice homographique chargée :")
print(homography_matrix)

# Charger la première frame de la vidéo
video_path = "input_videos/08fd33_4.mp4"  # Adapte ce chemin à ta vidéo
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise Exception(f"❌ Impossible de lire la vidéo : {video_path}")

# Points de référence initiaux (4 coins de la vidéo)
original_points = np.array([
    [0, 0],                                  # Coin haut gauche
    [frame.shape[1], 0],                     # Coin haut droit
    [0, frame.shape[0]],                     # Coin bas gauche
    [frame.shape[1], frame.shape[0]]         # Coin bas droit
], dtype="float32")

# Appliquer la transformation homographique
transformed_points = cv2.perspectiveTransform(original_points.reshape(-1, 1, 2), homography_matrix)
transformed_points = transformed_points.reshape(-1, 2)

# Dessiner les points transformés sur l'image
for i, (x, y) in enumerate(transformed_points):
    x, y = int(x), int(y)
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, f"P{i+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Calibration Check", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
