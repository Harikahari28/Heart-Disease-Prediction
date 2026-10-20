import cv2
import torch
import numpy as np

class EchoVideoProcessor:
    def __init__(self, clip_length=16, img_size=112):
        self.clip_length = clip_length
        self.img_size = img_size

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError("No frames found")

        return frames

    def sample_clip(self, frames):
        total = len(frames)
        if total < self.clip_length:
            frames += [frames[-1]] * (self.clip_length - total)
        else:
            start = (total - self.clip_length) // 2  # center clip
            frames = frames[start:start + self.clip_length]
        return frames

    def preprocess(self, video_path):
        frames = self.load_video(video_path)
        frames = self.sample_clip(frames)

        processed = []
        for f in frames:
            f = cv2.resize(f, (self.img_size, self.img_size))
            f = f.astype(np.float32) / 255.0
            processed.append(f)

        clip = np.stack(processed)                  # (T,H,W,C)
        clip = torch.tensor(clip).permute(3,0,1,2)  # (C,T,H,W)
        return clip.unsqueeze(0)                    # (1,C,T,H,W)
