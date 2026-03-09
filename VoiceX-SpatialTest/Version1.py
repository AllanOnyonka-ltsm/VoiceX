"""
VoiceX - Spatial Cough Tracker (Starter)
-----------------------------------------
Captures stereo audio → detects coughs → estimates direction via GCC-PHAT
→ tracks subjects → builds spatial timeline

Requirements:
    pip install sounddevice numpy scipy matplotlib
"""

import time
import threading
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

DEVICE_ID       = 1          # Microphone Array (Realtek) — change if needed
SAMPLE_RATE     = 16000      # Hz
CHUNK_DURATION  = 0.5        # seconds per audio chunk
CHUNK_SIZE      = int(SAMPLE_RATE * CHUNK_DURATION)

MIC_DISTANCE    = 0.06       # meters — laptop array ~6cm
SPEED_OF_SOUND  = 343        # m/s

# Cough detection thresholds
ENERGY_THRESHOLD     = 0.015   # RMS energy — tune this for your environment
MIN_COUGH_DURATION   = 0.08    # seconds — ignore very short bursts
COOLDOWN_SECONDS     = 1.0     # min gap between cough events per subject

# Subject clustering — coughs within this angle range = same subject
ANGLE_TOLERANCE = 20           # degrees


# ─────────────────────────────────────────
#  GCC-PHAT — Direction of Arrival
# ─────────────────────────────────────────

def gcc_phat(sig, refsig, fs):
    """
    Generalized Cross Correlation - Phase Transform
    Returns time delay (tau) in seconds between two mic signals.
    Negative = sound came from left, Positive = from right.
    """
    n = sig.shape[0] + refsig.shape[0]

    SIG    = np.fft.rfft(sig,    n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R  = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-10))

    max_shift = int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau   = shift / float(fs)

    return tau


def tau_to_angle(tau):
    """Convert time delay → angle in degrees. 0° = front, ±90° = sides."""
    ratio = np.clip((tau * SPEED_OF_SOUND) / MIC_DISTANCE, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))


def angle_to_direction(angle):
    if angle > 25:
        return "RIGHT"
    elif angle < -25:
        return "LEFT"
    else:
        return "CENTER"


# ─────────────────────────────────────────
#  COUGH DETECTOR — Energy Based
# ─────────────────────────────────────────

def is_cough(chunk_mono, fs=SAMPLE_RATE):
    """
    Simple energy-based cough detector.
    Later replace this with your CNN classifier.
    
    Returns (is_cough: bool, confidence: float)
    """
    rms = np.sqrt(np.mean(chunk_mono ** 2))

    if rms < ENERGY_THRESHOLD:
        return False, 0.0

    # Spectral check — coughs concentrate energy in 100Hz–3000Hz
    freqs, psd = signal.welch(chunk_mono, fs=fs, nperseg=256)
    cough_band  = np.sum(psd[(freqs >= 100) & (freqs <= 3000)])
    total_power = np.sum(psd) + 1e-10
    band_ratio  = cough_band / total_power

    # Simple confidence score from energy + band ratio
    confidence = float(np.clip(rms * 10 * band_ratio, 0, 1))

    is_cough_event = (rms > ENERGY_THRESHOLD) and (band_ratio > 0.4)

    return is_cough_event, confidence


# ─────────────────────────────────────────
#  SUBJECT TRACKER
# ─────────────────────────────────────────

class SubjectTracker:
    """
    Clusters cough events by angle → assigns subject IDs.
    Maintains timeline and per-subject risk aggregation.
    """

    def __init__(self):
        self.subjects      = {}          # subject_id → subject data
        self.timeline      = []          # flat list of all events
        self._next_id      = 1
        self._last_cough   = {}          # subject_id → last cough timestamp
        self._lock         = threading.Lock()

    def _find_subject(self, angle):
        """Find existing subject near this angle, or create new one."""
        for sid, data in self.subjects.items():
            if abs(data["angle"] - angle) <= ANGLE_TOLERANCE:
                # Update running average of angle (subject may shift slightly)
                data["angle"] = 0.8 * data["angle"] + 0.2 * angle
                return sid

        # New subject
        sid = f"Subject_{self._next_id}"
        self._next_id += 1
        self.subjects[sid] = {
            "angle":           angle,
            "direction":       angle_to_direction(angle),
            "cough_count":     0,
            "risk_scores":     [],
            "aggregated_risk": 0.0,
            "first_seen":      datetime.now(),
            "last_seen":       datetime.now(),
        }
        print(f"\n  [NEW SUBJECT] {sid} detected at {angle:.1f}° ({angle_to_direction(angle)})")
        return sid

    def register_cough(self, angle, confidence, timestamp=None):
        """Register a cough event and assign to a subject."""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            sid = self._find_subject(angle)

            # Cooldown check — avoid double-counting same cough
            last = self._last_cough.get(sid, 0)
            if timestamp - last < COOLDOWN_SECONDS:
                return None

            self._last_cough[sid] = timestamp

            # Update subject
            subj = self.subjects[sid]
            subj["cough_count"]    += 1
            subj["risk_scores"].append(confidence)
            subj["aggregated_risk"] = float(np.mean(subj["risk_scores"]))
            subj["last_seen"]       = datetime.now()

            # Log event
            event = {
                "subject":    sid,
                "timestamp":  timestamp,
                "angle":      angle,
                "direction":  angle_to_direction(angle),
                "confidence": confidence,
                "risk":       subj["aggregated_risk"],
            }
            self.timeline.append(event)

            return event

    def get_summary(self):
        with self._lock:
            return {
                sid: {
                    "angle":     d["angle"],
                    "direction": d["direction"],
                    "coughs":    d["cough_count"],
                    "risk":      d["aggregated_risk"],
                }
                for sid, d in self.subjects.items()
            }


# ─────────────────────────────────────────
#  VISUALIZER
# ─────────────────────────────────────────

class CoughVisualizer:
    """
    Real-time polar + timeline plots.
    Polar  → spatial position of subjects
    Timeline → cough events over time
    """

    def __init__(self, tracker: SubjectTracker):
        self.tracker = tracker
        plt.ion()

        self.fig = plt.figure(figsize=(14, 6))
        self.fig.suptitle("VoiceX — Spatial Cough Tracker", fontsize=14, fontweight="bold")

        # Left: polar spatial view
        self.ax_polar = self.fig.add_subplot(121, projection="polar")
        self.ax_polar.set_title("Spatial View\n(angle of cough source)", pad=15)

        # Right: timeline
        self.ax_time = self.fig.add_subplot(122)
        self.ax_time.set_title("Cough Timeline")
        self.ax_time.set_xlabel("Time (seconds)")
        self.ax_time.set_ylabel("Subject")

        self._start_time = time.time()
        self._colors     = plt.cm.tab10.colors

        plt.tight_layout()

    def _risk_color(self, risk):
        """Green (low) → Yellow → Red (high risk)."""
        r = float(np.clip(risk, 0, 1))
        return (r, 1 - r, 0)

    def update(self):
        summary  = self.tracker.get_summary()
        timeline = self.tracker.timeline[:]

        # ── Polar plot ───────────────────────────────
        self.ax_polar.clear()
        self.ax_polar.set_title("Spatial View\n(angle of cough source)", pad=15)
        self.ax_polar.set_theta_zero_location("N")   # 0° = top (front)
        self.ax_polar.set_theta_direction(-1)         # clockwise

        for i, (sid, data) in enumerate(summary.items()):
            theta = np.radians(data["angle"])
            r     = 0.6                               # fixed radius for clarity
            size  = 200 + data["coughs"] * 80
            color = self._risk_color(data["risk"])

            self.ax_polar.scatter(
                theta, r,
                s=size, c=[color], alpha=0.8,
                label=f"{sid} | {data['coughs']} coughs | risk {data['risk']:.2f}"
            )
            self.ax_polar.annotate(
                sid,
                (theta, r),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

        # Mic array marker at center
        self.ax_polar.scatter(0, 0, s=100, c="black", marker="^", zorder=5)
        self.ax_polar.set_ylim(0, 1)
        self.ax_polar.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)

        # ── Timeline plot ─────────────────────────────
        self.ax_time.clear()
        self.ax_time.set_title("Cough Timeline")
        self.ax_time.set_xlabel("Time (seconds from start)")
        self.ax_time.set_ylabel("Subject")

        subject_ids = list(summary.keys())

        for event in timeline:
            if event["subject"] not in subject_ids:
                continue
            y     = subject_ids.index(event["subject"])
            x     = event["timestamp"] - self._start_time
            color = self._risk_color(event["confidence"])

            self.ax_time.scatter(x, y, s=150, c=[color], alpha=0.85, zorder=3)
            self.ax_time.vlines(x, y - 0.1, y + 0.1, colors=[color], linewidth=2)

        if subject_ids:
            self.ax_time.set_yticks(range(len(subject_ids)))
            self.ax_time.set_yticklabels(subject_ids)

        self.ax_time.grid(axis="x", alpha=0.3)

        # Risk legend
        patches = [
            mpatches.Patch(color=(0, 1, 0), label="Low risk"),
            mpatches.Patch(color=(1, 1, 0), label="Medium risk"),
            mpatches.Patch(color=(1, 0, 0), label="High risk"),
        ]
        self.ax_time.legend(handles=patches, loc="upper right", fontsize=7)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("  VoiceX Spatial Cough Tracker")
    print("=" * 50)
    print(f"  Device  : {sd.query_devices(DEVICE_ID)['name']}")
    print(f"  Mic gap : {MIC_DISTANCE*100:.0f}cm")
    print(f"  Range   : ~{MIC_DISTANCE * 20 * 100:.0f}cm reliable")
    print(f"  Chunk   : {CHUNK_DURATION*1000:.0f}ms")
    print("\n  Make cough sounds — move left/right to test direction")
    print("  Press Ctrl+C to stop\n")

    tracker    = SubjectTracker()
    visualizer = CoughVisualizer(tracker)

    def audio_callback(indata, frames, time_info, status):
        """Called every CHUNK_DURATION seconds with stereo audio."""
        if status:
            print(f"  [AUDIO WARNING] {status}")

        # Split stereo channels
        mic1 = indata[:, 0].copy()
        mic2 = indata[:, 1].copy() if indata.shape[1] > 1 else indata[:, 0].copy()
        mono = (mic1 + mic2) / 2

        # Step 1 — Cough detection
        detected, confidence = is_cough(mono)
        if not detected:
            return

        # Step 2 — Direction estimation
        tau   = gcc_phat(mic1, mic2, SAMPLE_RATE)
        angle = tau_to_angle(tau)

        # Step 3 — Register with tracker
        event = tracker.register_cough(angle, confidence)
        if event:
            print(
                f"  [COUGH] {event['subject']:12s} | "
                f"angle: {angle:+6.1f}° {event['direction']:6s} | "
                f"conf: {confidence:.2f} | "
                f"risk: {event['risk']:.2f}"
            )

    # Start audio stream
    with sd.InputStream(
        device=DEVICE_ID,
        channels=2,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        try:
            while True:
                visualizer.update()
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n  Stopped. Final summary:\n")
            summary = tracker.get_summary()
            for sid, data in summary.items():
                print(
                    f"  {sid:12s} | angle: {data['angle']:+6.1f}° "
                    f"| coughs: {data['coughs']:3d} "
                    f"| risk: {data['risk']:.2f} "
                    f"| {data['direction']}"
                )
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()