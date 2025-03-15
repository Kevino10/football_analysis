import cv2
import numpy as np
from utils import get_center_of_bbox

def draw_ellipse(frame, bbox, color, track_id=None):
    if len(bbox) != 4:
        return frame

    x, _ = get_center_of_bbox(bbox)
    y = int(bbox[3])
    width = int(bbox[2] - bbox[0])

    cv2.ellipse(
        frame,
        (int(x), int(y)),
        (max(1, width // 2), 10),
        0,
        -45,
        235,
        color,
        2
    )

    if track_id is not None:
        cv2.putText(frame, f"ID: {track_id}", (int(x), int(y) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame

def draw_triangle(frame, bbox, color):
    x, y = get_center_of_bbox(bbox)
    points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])
    cv2.drawContours(frame, [points], 0, color, -1)
    cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)

def draw_annotations(video_frames, tracks, team_ball_control):
    output_frames = []

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        players = tracks['players'][frame_num]
        referees = tracks.get('referees', [[]])
        if frame_num < len(referees):
            referees = referees[frame_num]
        else:
            referees = []

        ball = tracks['ball'][frame_num]

        # Log de contrôle
        print(f"Frame {frame_num} - {len(players)} joueurs - {len(referees)} arbitres - Ballon : {bool(ball)}")

        # Dessin des joueurs
        for player in players:
            draw_ellipse(frame, player['bbox'], (0, 0, 255), player['id'])

        # Dessin des arbitres
        for referee in referees:
            draw_ellipse(frame, referee['bbox'], (0, 255, 255))

        # Dessin du ballon
        if ball:
            draw_triangle(frame, ball['bbox'], (0, 255, 0))

        output_frames.append(frame)

    return output_frames
