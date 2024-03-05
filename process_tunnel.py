# Author: Andy Lustig
# Created: 2024-03-06

import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Capture:
    def __init__(self, video_path):
        cap = cv2.VideoCapture(video_path)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.cap = cap


class RatLabel:
    def __init__(self, label_data, frame_times):
        self.data = label_data

        times, lefts, centers, rights = [], [], [], []

        for frame in self.data["sequence"]:
            x, width = frame["x"], frame["width"]
            times.append(frame["time"])
            lefts.append(x)
            centers.append(x + width / 2)
            rights.append(x + width)

        self.lefts = np.interp(frame_times, times, lefts)
        self.centers = np.interp(frame_times, times, centers)
        self.rights = np.interp(frame_times, times, rights)

        self.times = np.array(times)
        self.frame_times = frame_times


def create_interpolated_arrays(input_json, capture):
    frame_count = capture.frame_count
    fps = capture.fps
    # read json annotation data
    with open(input_json, "r") as annotations:
        data = json.load(annotations)
    a_labels, b_labels = data[0]["box"]

    # create data arrays for plotting
    frame_times = np.arange(0, frame_count, 1 / fps)
    rat_a = RatLabel(a_labels, frame_times)
    rat_b = RatLabel(b_labels, frame_times)

    return rat_a, rat_b


def create_animation(rat_a, rat_b, capture, outtput_name):
    FILL = True

    X_LABEL = "Time (s)"
    Y_LABEL = "Distance from left wall\n(% tunnel width)"
    plt.style.use("dark_background")

    print("Creating animation...")
    frame_count = capture.frame_count
    fps = capture.fps
    width = capture.width
    height = capture.height

    dpi = 300
    width = width / dpi
    height = height / dpi
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ft = rat_a.frame_times
    pos_a = rat_a.rights
    pos_b = rat_b.lefts

    line_alpha, fill_alpha = 0.7, 0.4

    ax.plot(ft[0], pos_a[0], color="#1f77b4", label="A", alpha=line_alpha)
    ax.plot(ft[0], pos_b[0], color="#ff7f0c", label="B", alpha=line_alpha)
    if FILL:
        ax.fill_between(ft[0], [0], pos_a[0], color="#1f77b4", alpha=fill_alpha)
        ax.fill_between(ft[0], [100], pos_b[0], color="#ff7f0c", alpha=fill_alpha)

    ax.set(
        xlim=[0, frame_count / fps],
        ylim=[0, 100],
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
    )

    def draw_frame(frame):
        if frame % (int(frame_count / 10)) == 1:
            print(f"{frame/frame_count:.0%}")
        ax.cla()
        ax.plot(ft[:frame], pos_a[:frame], color="#1f77b4", label="A", alpha=line_alpha)
        ax.plot(ft[0], pos_b[0], color="#ff7f0c", label="B", alpha=line_alpha)

        if FILL:
            ax.fill_between(
                ft[:frame],
                np.zeros(frame),
                pos_a[:frame],
                color="#1f77b4",
                alpha=fill_alpha,
            )
            ax.fill_between(
                ft[:frame],
                np.ones(frame) * 100,
                pos_b[:frame],
                color="#ff7f0c",
                alpha=fill_alpha,
            )

        ax.set(
            xlim=[0, frame_count / fps],
            ylim=[0, 100],
            xlabel=X_LABEL,
            ylabel=Y_LABEL,
        )
        plt.tight_layout()

    # plt.legend()
    my_animation = animation.FuncAnimation(
        fig=fig,
        func=draw_frame,
        frames=frame_count,
        interval=1000 / fps,
    )
    my_animation.save(
        outtput_name,
        fps=fps,
        extra_args=["-vcodec", "libx264"],
    )

    print(f"...animation saved as {outtput_name}\n")


def make_marked_video(rat_a, rat_b, capture, filename):
    print("Creating annotated video...")
    fps = capture.fps
    width = capture.width
    height = capture.height
    cap = capture.cap

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    frame_num = 0

    BLUE = (180, 119, 31)
    ORANGE = (12, 127, 255)

    vcenter = int(height / 2)
    bar_offset = 220
    bar_thickness = 40

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.rectangle(
            frame,
            (0, vcenter - bar_offset),
            (
                int(rat_a.rights[frame_num] / 100 * width),
                vcenter - (bar_thickness + bar_offset),
            ),
            BLUE,
            -1,
        )

        cv2.rectangle(
            frame,
            (width, vcenter - bar_offset),
            (
                int(rat_b.lefts[frame_num] / 100 * width),
                vcenter - (bar_thickness + bar_offset),
            ),
            ORANGE,
            -1,
        )

        cv2.imshow("RatTube", frame)
        vid_writer.write(frame)

        # Check for the 'q' key press to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        frame_num += 1

    cap.release()
    vid_writer.release()

    cv2.destroyAllWindows()

    print(f"...marked video saved as {filename}")


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A script that takes in raw video and labels and produces an annotated video and an animated plot of the label positions over time"
    )

    # Add arguments
    parser.add_argument("footage", type=str, help="Original video footage")
    parser.add_argument("labels", type=str, help="JSON-MIN file from Label Studio")

    # Parse the arguments
    args = parser.parse_args()

    capture = Capture(args.footage)

    if not os.path.exists("output"):
        os.makedirs("output")

    rat_a, rat_b = create_interpolated_arrays(args.labels, capture)
    create_animation(rat_a, rat_b, capture, "output/plot.mp4")
    make_marked_video(rat_a, rat_b, capture, "output/marked.mp4")


if __name__ == "__main__":
    main()
