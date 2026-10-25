import time
from picarx import Picarx

car = Picarx()
car.stop()

try:
    for angle in (0, 40, -40):
        car.set_cam_pan_angle(angle)
        time.sleep(0.6)

        print(f"\nPan servo command: {angle:+d}°", flush=True)

        readings = []
        for _ in range(5):
            distance = float(car.get_distance())

            if distance > 0:
                readings.append(round(distance, 1))

            time.sleep(0.1)

        print(f"Distance readings (cm): {readings}", flush=True)

        input("Observe the actual pan direction, then press Enter to continue...")

finally:
    car.stop()
    car.set_cam_pan_angle(0)