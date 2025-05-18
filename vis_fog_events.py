import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_fog_data(folder_path):
    from collections import defaultdict
    all_data = defaultdict(list)  # subject_id -> list of rows
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            subject_id = file_name[:3]  # e.g., 'S01'
            with open(file_path, 'r') as file:
                for line in file:
                    values = line.strip().split()
                    values = [float(v) for v in values]
                    all_data[subject_id].append(values)
    # Convert lists to numpy arrays
    for subject_id in all_data.keys():
        all_data[subject_id] = np.array(all_data[subject_id], dtype=np.float32)
    return all_data

def extract_segments(data, min_length=128):
    """
    Given the raw data (a NumPy array) for one subject, this function:
      - Discards rows with annotation==0.
      - Finds contiguous segments where the annotation is constant.
      - Returns two lists: one for non-freezing segments (annotation==1)
        and one for freezing segments (annotation==2).
    Only segments with at least min_length samples are returned.
    """
    # Filter out rows with annotation 0
    data = data[data[:, -1] != 0]
    if len(data) == 0:
        return [], []
    
    # Get the annotation column (assumed to be the last column)
    annotations = data[:, -1]
    segments_nonfog = []
    segments_fog = []
    start_idx = 0

    for i in range(1, len(annotations)):
        # When the annotation changes, we have reached the end of a segment.
        if annotations[i] != annotations[i - 1]:
            segment = data[start_idx:i, :]
            if len(segment) >= min_length:
                if np.mean(segment[:, -1]) < 1.5:
                    segments_nonfog.append(segment)
                else:
                    segments_fog.append(segment)
            start_idx = i
    # Add last segment
    segment = data[start_idx:, :]
    if len(segment) >= min_length:
        if np.mean(segment[:, -1]) < 1.5:
            segments_nonfog.append(segment)
        else:
            segments_fog.append(segment)
    return segments_nonfog, segments_fog

def plot_segments(nonfog_segment, fog_segment, channel=1):
    """
    Plots the specified channel (default channel 1) for two segments:
      - nonfog_segment: segment with annotation ~1 (no freezing)
      - fog_segment: segment with annotation ~2 (freezing event)
    Assumes the first column is time (in ms) and that the channel indices are relative
    to the full data matrix.
    """
    plt.figure(figsize=(12, 6))

    # Plot non-freezing segment
    plt.subplot(2, 1, 1)
    plt.plot(nonfog_segment[:, 0], nonfog_segment[:, channel], color='blue')
    plt.title("Non-Freezing Segment (Annotation ~1)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (mg)")

    # Plot freezing segment
    plt.subplot(2, 1, 2)
    plt.plot(fog_segment[:, 0], fog_segment[:, channel], color='red')
    plt.title("Freezing Segment (Annotation ~2)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (mg)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FoG vs Non-FoG events")
    parser.add_argument('--subject', type=str, default=None, help="Subject ID to visualize (e.g., S01). If not provided, one will be selected at random.")
    parser.add_argument('--folder', type=str, default='fog_data', help="Folder containing the fog .txt files")
    parser.add_argument('--min_length', type=int, default=128, help="Minimum length of segment to consider")
    parser.add_argument('--channel', type=int, default=1, help="Index of the sensor channel to plot (e.g., 1 for the first sensor channel)")
    args = parser.parse_args()

    # Load data
    all_data = load_fog_data(args.folder)
    available_subjects = list(all_data.keys())
    if args.subject is None:
        # Choose a random subject if not provided
        subject_id = np.random.choice(available_subjects)
    else:
        subject_id = args.subject
    print(f"Visualizing subject: {subject_id}")
    subject_data = all_data[subject_id]

    # Extract contiguous segments
    segments_nonfog, segments_fog = extract_segments(subject_data, min_length=args.min_length)
    
    if len(segments_nonfog) == 0:
        print("No non-freezing segments found.")
    if len(segments_fog) == 0:
        print("No freezing segments found.")
    
    # For visualization, pick the first segment from each list (or random if you prefer)
    if len(segments_nonfog) > 0 and len(segments_fog) > 0:
        nonfog_segment = segments_nonfog[0]
        fog_segment = segments_fog[0]
        plot_segments(nonfog_segment, fog_segment, channel=args.channel)
    else:
        print("Insufficient segments for visualization.")
