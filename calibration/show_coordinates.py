import cv2

def show_video_with_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception(f"❌ Impossible de lire la vidéo : {video_path}")

    # Fonction callback pour afficher les coordonnées dynamiques
    def show_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            temp_frame = frame.copy()
            # Affiche les coordonnées en haut à gauche de la fenêtre
            cv2.putText(temp_frame, f"({x}, {y})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Coordonnées - Survolez et notez", temp_frame)

    cv2.imshow("Coordonnées - Survolez et notez", frame)
    cv2.setMouseCallback("Coordonnées - Survolez et notez", show_coordinates)

    print("🔎 Survolez la vidéo pour voir les coordonnées x, y.")
    print("📋 Notez les 4 points nécessaires pour la calibration (ex: corners visibles).")
    print("Appuyez sur une touche pour fermer la fenêtre une fois que vous avez les coordonnées.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Lancement
if __name__ == "__main__":
    show_video_with_coordinates("calibration/tmp_frame_0.jpg")
