from qorageom import detector_pose, pixel_to_q_lab


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
