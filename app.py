# ========================= TRAFFIC VIDEO AI APP =========================
import streamlit as st
import os, cv2, tempfile, shutil, uuid, datetime
import torch
import numpy as np
from ultralytics import YOLO
from collections import Counter

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="🚦 Traffic Analysis Video AI", layout="wide")
st.title("🚦 Traffic Analysis (Video AI)")

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES     = 200
TARGET_FPS     = 2
CONF_THRES     = 0.30
MAX_VIOLATIONS = 20
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

# ── Accident heuristic thresholds ──────────────────────────────────────
# WHY these values:
#   Normal traffic perspective causes apparent IoU of 0.05–0.25 between
#   cars in adjacent lanes. Real physical collisions merge boxes deeply.
OVERLAP_IOU_THRESH     = 0.40  # deep bbox merge only (not perspective illusion)
OVERLAP_PERSIST_FRAMES = 2     # overlap must hold N consecutive sampled frames
CLUSTER_DIST_THRESH    = 40    # very tight centroid distance (px) to avoid traffic-jam FP
CLUSTER_MIN_VEHICLES   = 3     # need 3+ vehicles in cluster
SUDDEN_STOP_THRESH     = 0.70  # vehicle count must drop 70%+ vs rolling avg
MOTION_SPIKE_THRESH    = 60.0  # normal traffic ~20–35px/frame; accidents spike above 60
MIN_SIGNALS            = 2     # require this many independent signals per incident
DEDUP_FRAME_GAP        = 4     # ignore new accident within N frames of last one

# -----------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    m = YOLO("yolov8n.pt")
    m.to(DEVICE)
    return m

yolo = load_yolo()

# -----------------------------------------------------------------------
# SESSION DIR
# -----------------------------------------------------------------------
def make_session_dir():
    sid  = st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    path = os.path.join(tempfile.gettempdir(), f"traffic_{sid}")
    os.makedirs(path, exist_ok=True)
    return path

# -----------------------------------------------------------------------
# FRAME EXTRACTION
# -----------------------------------------------------------------------
def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if not raw_fps or raw_fps <= 0:
        st.warning("Could not detect FPS — defaulting to 30.")
        raw_fps = 30.0
    interval = max(1, int(raw_fps // TARGET_FPS))
    frames, idx, saved = [], 0, 0
    while cap.isOpened() and saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            p = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(p, frame)
            frames.append(p)
            saved += 1
        idx += 1
    cap.release()
    return frames, raw_fps

# -----------------------------------------------------------------------
# DETECTION
# -----------------------------------------------------------------------
def detect_objects(img_path):
    res  = yolo(img_path, conf=CONF_THRES, verbose=False)[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        dets.append({
            "label": res.names[int(b.cls[0])],
            "conf":  float(b.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })
    return dets

# -----------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------
def iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (aa + ab - inter)

def cdist(a, b):
    return ((a["cx"] - b["cx"])**2 + (a["cy"] - b["cy"])**2) ** 0.5

# -----------------------------------------------------------------------
# ACCIDENT HEURISTICS
# -----------------------------------------------------------------------
def check_overlap(dets):
    """
    Only fires when IoU >= 0.40 — eliminates the perspective-overlap false
    positives that normal adjacent-lane traffic causes (typical IoU 0.05–0.25).
    """
    veh = [d for d in dets if d["label"] in VEHICLE_CLASSES]
    for i in range(len(veh)):
        for j in range(i + 1, len(veh)):
            v = iou(veh[i], veh[j])
            if v >= OVERLAP_IOU_THRESH:
                return True, f"{veh[i]['label']} ↔ {veh[j]['label']} deep overlap (IoU {v:.2f})"
    return False, ""

def check_cluster(dets):
    """
    Requires CLUSTER_MIN_VEHICLES vehicles within CLUSTER_DIST_THRESH px.
    Tighter than before to avoid flagging normal gridlock at traffic lights.
    """
    veh = [d for d in dets if d["label"] in VEHICLE_CLASSES]
    if len(veh) < CLUSTER_MIN_VEHICLES:
        return False, ""
    for i in range(len(veh)):
        close = [veh[j] for j in range(len(veh))
                 if i != j and cdist(veh[i], veh[j]) < CLUSTER_DIST_THRESH]
        if len(close) >= CLUSTER_MIN_VEHICLES - 1:
            return True, f"{len(close)+1} vehicles within {CLUSTER_DIST_THRESH}px"
    return False, ""

def check_sudden_stop(current, rolling_avg):
    if rolling_avg > 2 and current < rolling_avg * (1 - SUDDEN_STOP_THRESH):
        return True, f"Count dropped from ~{rolling_avg:.1f} to {current}"
    return False, ""

def check_motion(prev_gray, curr_gray):
    if prev_gray is None or curr_gray is None:
        return False, "", 0.0
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        if mag > MOTION_SPIKE_THRESH:
            return True, f"Motion spike {mag:.1f}px/frame", mag
        return False, "", mag
    except Exception:
        return False, "", 0.0

# -----------------------------------------------------------------------
# ANNOTATE FRAME
# -----------------------------------------------------------------------
def annotate_frame(frame_path, reasons, dets):
    img = cv2.imread(frame_path)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    for d in dets:
        if d["label"] in VEHICLE_CLASSES:
            cv2.rectangle(img,
                          (int(d["x1"]), int(d["y1"])),
                          (int(d["x2"]), int(d["y2"])),
                          (0, 0, 255), 2)
            cv2.putText(img, d["label"],
                        (int(d["x1"]), max(0, int(d["y1"]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 180), -1)
    cv2.putText(img, "*** ACCIDENT DETECTED ***", (8, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    for k, r in enumerate(reasons[:4]):
        cv2.putText(img, f"• {r}", (8, 60 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)
    return img

# -----------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------
def build_report(video_name, total_frames, fps, congestion_data,
                 label_counter, accidents, violations):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = [
        "=" * 60,
        "       TRAFFIC INCIDENT ANALYSIS REPORT",
        "=" * 60,
        f"Generated  : {now}",
        f"Video File : {video_name}",
        f"Source FPS : {fps:.1f}  |  Frames Analysed : {total_frames}",
        "",
        "── SUMMARY ─────────────────────────────────────────────",
        f"  Average Congestion  : {np.mean(congestion_data):.2f} vehicles/frame",
        f"  Peak Congestion     : {int(np.max(congestion_data))} vehicles/frame",
        f"  Accidents Detected  : {len(accidents)}",
        f"  Other Violations    : {len(violations)}",
        "",
        "── OBJECT COUNTS ────────────────────────────────────────",
    ]
    for label, count in label_counter.most_common():
        L.append(f"  {label:<22}: {count}")
    L += ["", "── ACCIDENT EVENTS ──────────────────────────────────────"]
    if accidents:
        for i, acc in enumerate(accidents, 1):
            L += [
                "",
                f"  Incident #{i}",
                f"  Frame Index : {acc['frame_idx']}",
                f"  Frame File  : {os.path.basename(acc['frame_path'])}",
                f"  Triggers    :",
            ]
            for r in acc["reasons"]:
                L.append(f"    • {r}")
    else:
        L.append("  None detected.")
    L += ["", "── OTHER VIOLATIONS ─────────────────────────────────────"]
    if violations:
        for fp, vtype in violations:
            L.append(f"  {vtype:<30} | {os.path.basename(fp)}")
    else:
        L.append("  None detected.")
    L += ["", "=" * 60, "END OF REPORT", "=" * 60]
    return "\n".join(L)

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def congestion_score(dets):
    return sum(1 for d in dets if d["label"] in VEHICLE_CLASSES)

def triple_riding(dets):
    persons = sum(1 for d in dets if d["label"] == "person")
    bikes   = sum(1 for d in dets if d["label"] == "motorcycle")
    return bikes == 1 and persons >= 3

# ======================================================================
# MAIN UI
# ======================================================================
video_file = st.file_uploader(
    "Upload traffic video (≤60 s recommended)", type=["mp4", "avi", "mov"]
)

if video_file:
    frames_dir    = make_session_dir()
    annotated_dir = os.path.join(frames_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=frames_dir) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("Extracting frames…")
    try:
        frames, detected_fps = extract_frames(video_path, frames_dir)
    except Exception as e:
        st.error(f"Frame extraction failed: {e}"); st.stop()

    if not frames:
        st.error("No frames could be extracted. Is the file valid?"); st.stop()

    st.success(f"Extracted **{len(frames)}** frames  |  Source FPS: {detected_fps:.1f}")

    # ── PROCESSING ────────────────────────────────────────────────────
    congestion_data = []
    violations      = []
    accidents       = []
    label_counter   = Counter()
    flow_data       = []
    errors          = []
    prev_gray       = None
    count_window    = []
    overlap_streak  = 0   # tracks consecutive frames with deep vehicle overlap

    st.subheader("🔍 Processing Frames")
    prog = st.progress(0)

    for i, frame_path in enumerate(frames):
        try:
            dets = detect_objects(frame_path)
        except Exception as e:
            errors.append(f"Frame {i}: {e}")
            congestion_data.append(0)
            flow_data.append(0.0)
            overlap_streak = 0
            prog.progress((i + 1) / len(frames))
            continue

        for d in dets:
            label_counter[d["label"]] += 1

        v_count = congestion_score(dets)
        congestion_data.append(v_count)
        count_window.append(v_count)
        if len(count_window) > 10:
            count_window.pop(0)
        rolling_avg = float(np.mean(count_window))

        # optical flow
        bgr       = cv2.imread(frame_path)
        curr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None else None
        motion_flag, motion_reason, flow_mag = check_motion(prev_gray, curr_gray)
        flow_data.append(flow_mag)
        prev_gray = curr_gray

        # ── ACCIDENT SIGNALS ──────────────────────────────────────────
        reasons = []

        # Signal 1: Deep overlap — must persist across consecutive frames
        # (single-frame overlap is almost always a perspective artefact)
        overlap_now, overlap_reason = check_overlap(dets)
        if overlap_now:
            overlap_streak += 1
        else:
            overlap_streak = 0
        if overlap_streak >= OVERLAP_PERSIST_FRAMES:
            reasons.append(
                f"Sustained vehicle overlap — {overlap_reason} "
                f"({overlap_streak} consecutive frames)"
            )

        # Signal 2: Tight vehicle cluster
        f, r = check_cluster(dets)
        if f:
            reasons.append(f"Vehicle cluster — {r}")

        # Signal 3: Sharp vehicle count drop
        f, r = check_sudden_stop(v_count, rolling_avg)
        if f:
            reasons.append(f"Sudden stop — {r}")

        # Signal 4: Chaotic optical flow spike
        if motion_flag:
            reasons.append(f"Motion spike — {motion_reason}")

        # Fire only when ≥ MIN_SIGNALS triggered AND not a duplicate of last event
        last_acc_frame = accidents[-1]["frame_idx"] if accidents else -99
        is_new_event   = (i - last_acc_frame) > DEDUP_FRAME_GAP
        if len(reasons) >= MIN_SIGNALS and is_new_event and len(accidents) < MAX_VIOLATIONS:
            ann_img  = annotate_frame(frame_path, reasons, dets)
            ann_path = os.path.join(
                annotated_dir, f"accident_{len(accidents):03d}_f{i:04d}.jpg"
            )
            cv2.imwrite(ann_path, ann_img)
            accidents.append({
                "frame_idx":  i,
                "frame_path": frame_path,
                "ann_path":   ann_path,
                "reasons":    reasons,
            })

        # Other violations
        if triple_riding(dets) and len(violations) < MAX_VIOLATIONS:
            violations.append((frame_path, "Triple Riding"))

        prog.progress((i + 1) / len(frames))

    if errors:
        with st.expander(f"⚠️ {len(errors)} frame(s) failed"):
            st.write("\n".join(errors[:20]))

    # ── REPORT ────────────────────────────────────────────────────────
    report_text = build_report(
        video_file.name, len(frames), detected_fps,
        congestion_data, label_counter, accidents, violations
    )

    # ── DASHBOARD ─────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Frames",       len(frames))
    c2.metric("Avg Congestion",     round(float(np.mean(congestion_data)), 2))
    c3.metric("Peak Congestion",    int(np.max(congestion_data)))
    c4.metric("Accidents Detected", len(accidents))
    c5.metric("Other Violations",   len(violations))

    # ── ACCIDENT SECTION ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Accident Detection Results")
    if accidents:
        st.error(f"**{len(accidents)} accident event(s) detected.**")
        for k, acc in enumerate(accidents):
            with st.expander(
                f"Incident #{k+1}  —  Frame {acc['frame_idx']}", expanded=(k == 0)
            ):
                st.image(acc["ann_path"],
                         caption=f"Annotated Frame {acc['frame_idx']}",
                         use_container_width=True)
                st.markdown("**Triggered signals:**")
                for r in acc["reasons"]:
                    st.markdown(f"- {r}")
                with open(acc["ann_path"], "rb") as fh:
                    st.download_button(
                        label     = f"⬇️ Download Annotated Frame #{k+1}",
                        data      = fh.read(),
                        file_name = os.path.basename(acc["ann_path"]),
                        mime      = "image/jpeg",
                        key       = f"dl_frame_{k}",
                    )
    else:
        st.success("No accident events detected.")

    # ── REPORT DOWNLOAD ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Full Incident Report")
    st.text(report_text)
    st.download_button(
        label     = "⬇️ Download Report (.txt)",
        data      = report_text.encode("utf-8"),
        file_name = f"traffic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime      = "text/plain",
    )

    # ── OBJECT COUNTS ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚗 Object Frequency")
    traffic_labels = {k: v for k, v in label_counter.items()
                      if k in VEHICLE_CLASSES | {"person"}}
    other_labels   = {k: v for k, v in label_counter.items()
                      if k not in traffic_labels}
    for k, v in sorted(traffic_labels.items(), key=lambda x: -x[1]):
        st.write(f"- **{k}** : {v}")
    if other_labels:
        with st.expander("Other detected objects"):
            for k, v in sorted(other_labels.items(), key=lambda x: -x[1]):
                st.write(f"- {k} : {v}")

    # ── OTHER VIOLATIONS ──────────────────────────────────────────────
    st.subheader("🚦 Other Violations")
    if violations:
        cols = st.columns(min(3, len(violations[:6])))
        for idx, (fp, vtype) in enumerate(violations[:6]):
            cols[idx % 3].image(fp, caption=vtype, use_container_width=True)
    else:
        st.success("No other violations detected.")

    # ── CHARTS ────────────────────────────────────────────────────────
    st.subheader("📈 Congestion Over Time")
    st.line_chart(congestion_data)
    st.subheader("🌊 Motion Intensity (Optical Flow)")
    st.line_chart(flow_data)

    # ── CLEANUP ───────────────────────────────────────────────────────
    try:
        shutil.rmtree(frames_dir)
        del st.session_state["session_id"]
    except Exception:
        pass

else:
    st.info("Upload a traffic video to begin analysis.")
    st.caption(
        f"Running on: **{DEVICE.upper()}**  |  "
        f"Max frames: {MAX_FRAMES}  |  Sampling: {TARGET_FPS} fps"
    )
