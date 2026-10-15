import numpy as np
from math import tau

HC_KEV_ANGSTROM = 12.398


def rotate(vector, axis, angle_deg):
    v = np.asarray(vector, dtype=float)
    a = np.array(axis, dtype=float, copy=True)
    a /= np.linalg.norm(a)

    angle = np.deg2rad(angle_deg)
    return (
        v * np.cos(angle)
        + np.cross(a, v) * np.sin(angle)
        + a * np.dot(a, v) * (1 - np.cos(angle))
    )


def pixel_to_q_lab(
    energy_kev,
    incident_dir,
    center_pos,
    pixel_x_dir,
    pixel_y_dir,
    pixel_size_m,
    px,
    py,
):
    """검출기 중심 기준 픽셀 (px, py)의 q를 실험실 좌표로 계산한다."""
    incident = np.asarray(incident_dir, dtype=float)
    incident = incident / np.linalg.norm(incident)

    pixel_pos = (
        np.asarray(center_pos, dtype=float)
        + pixel_size_m * px * np.asarray(pixel_x_dir, dtype=float)
        + pixel_size_m * py * np.asarray(pixel_y_dir, dtype=float)
    )

    k = tau * energy_kev / HC_KEV_ANGSTROM  # [Å^-1]
    ki = k * incident
    kf = k * pixel_pos / np.linalg.norm(pixel_pos)
    return kf - ki


def detector_pose(
    distance_m,
    incident_dir,
    scatter_up,
    up_angle_deg,
    side_angle_deg,
):
    """예제 장비의 각도에서 검출기 중심과 두 픽셀 방향을 구한다."""
    incident = np.asarray(incident_dir, dtype=float)
    up = np.asarray(scatter_up, dtype=float)
    incident = incident / np.linalg.norm(incident)
    up = up / np.linalg.norm(up)

    if not np.isclose(np.dot(incident, up), 0.0):
        raise ValueError("incident_dir and scatter_up must be perpendicular")

    side = np.cross(up, incident)

    def turn(v):
        v = rotate(v, -side, up_angle_deg)
        return rotate(v, up, side_angle_deg)

    center_pos = distance_m * turn(incident)
    pixel_x_dir = turn(side)
    pixel_y_dir = turn(up)
    return center_pos, pixel_x_dir, pixel_y_dir


def main():
    incident = [1, 0, 0]
    up = [0, 1, 0]

    center_pos, pixel_x_dir, pixel_y_dir = detector_pose(
        distance_m=1.0,
        incident_dir=incident,
        scatter_up=up,
        up_angle_deg=30,
        side_angle_deg=0,
    )

    q = pixel_to_q_lab(
        energy_kev=8.04,
        incident_dir=incident,
        center_pos=center_pos,
        pixel_x_dir=pixel_x_dir,
        pixel_y_dir=pixel_y_dir,
        pixel_size_m=55e-6,
        px=1,
        py=1,
    )

    print("center_pos:  ", center_pos)
    print("pixel_x_dir:", pixel_x_dir)
    print("pixel_y_dir:", pixel_y_dir)
    print("q_lab:      ", q)


if __name__ == "__main__":
    main()
