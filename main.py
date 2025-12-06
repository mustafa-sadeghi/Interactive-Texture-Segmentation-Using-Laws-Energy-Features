"""
================================================================================
Machine Vision Course                    MiniProject-7&8: Laws' Texture Segmentation
================================================================================
Professor          : Prof. Hamidreza Pourreza
Institution        : Ferdowsi University of Mashhad
Term               : Spring 2025

Student Name       : Mustafa Sadeghi
Student ID         : 4027390423

Delivery Deadline  : 2025-09-06
Delivery Date      : 2025-09-07

Description:
    This script implements an texture segmentation Algorithm
    based on Laws’ energy features. The pipeline:
      1) Preprocessing:
         - Input grayscale image normalization.
      2) Feature extraction:
         - 16 Laws masks → reduced to 9 energy maps.
         - Pointwise or fragmental feature representation.
      3) Prototype-based segmentation:
         - User selects ROIs interactively (per class).
         - Prototypes extracted via median of ROI features.
         - Per-class acceptance thresholds (δ-percentile).
      4) Segmentation:
         - Class assignment by minimum distance to prototypes.
         - Reject pixels exceeding δ threshold.
      5) Outputs:
         - Saved plots: energy maps, ROI overlays, segmentation results.
         - Console reports number of classified pixels.
         - Interactive OpenCV window for ROI definition.

Controls:
    - Keys [1..9] (ASCII / Persian / Arabic-Indic): switch class
    - Left-click: add centered square ROI
    - +/- : increase/decrease ROI size
    - Backspace: remove last ROI
    - R: reset all ROIs
    - Enter: run segmentation
    - Esc: quit

Dependencies:
    - OpenCV        (pip install opencv-python)
    - NumPy         (pip install numpy)
    - SciPy         (pip install scipy)
    - scikit-learn  (pip install scikit-learn)
    - Matplotlib    (pip install matplotlib)

Author            : Mustafa Sadeghi
================================================================================
"""


import os
import numpy as np
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ---------- High-contrast palette ----------
PALETTE_RGB = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (255, 127, 0),
    (152, 78, 163),
    (255, 255, 51),
    (166, 86, 40),
    (247, 129, 191),
    (153, 153, 153),
]
ROI_COLORS_BGR = [(b, g, r) for (r, g, b) in PALETTE_RGB]

# ---------- Laws-based segmenter ----------
class LawsTextureSegmentation:
    def __init__(self, energy_window_size=15, norm_window_size=15):
        self.energy_window_size = energy_window_size
        self.norm_window_size = norm_window_size
        self.scaler = StandardScaler()
        self._generate_masks()
        self.energy9_std = None

    def _generate_masks(self):
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        R5 = np.array([1, -4, 6, -4, 1])
        vectors = [L5, E5, S5, R5]
        names = ['L5', 'E5', 'S5', 'R5']
        self.masks, self.mask_pairs = [], []
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                self.masks.append(np.outer(v1, v2))
                self.mask_pairs.append(f"{names[i]}{names[j]}")
        self.masks = np.asarray(self.masks)

    def _normalize_intensity(self, image):
        local_mean = ndimage.uniform_filter(image.astype(np.float64),
                                            size=self.norm_window_size, mode='reflect')
        return image.astype(np.float64) - local_mean

    def _apply_masks(self, image):
        return np.asarray([ndimage.convolve(image, k, mode='reflect') for k in self.masks])

    def _energy_sum(self, filt_stack):
        kernel = np.ones((self.energy_window_size, self.energy_window_size), dtype=np.float64)
        return np.asarray([ndimage.convolve(np.abs(fr), kernel, mode='reflect') for fr in filt_stack])

    def _combine_to_9(self, energy16):
        idx = {name: i for i, name in enumerate(self.mask_pairs)}
        pairs = [('L5E5','E5L5'), ('L5S5','S5L5'), ('L5R5','R5L5'),
                 ('E5S5','S5E5'), ('E5R5','R5E5'), ('S5R5','R5S5')]
        singles = ['E5E5','S5S5','R5R5']  # drop L5L5
        out = [0.5*(energy16[idx[a]] + energy16[idx[b]]) for a,b in pairs]
        out += [energy16[idx[s]] for s in singles]
        return np.stack(out, axis=0)

    def extract_features(self, image, mode='pointwise', fragment_size=15, fit_scaler=True):
        assert mode in ('pointwise', 'fragmental')
        img = self._normalize_intensity(image)
        energy16 = self._energy_sum(self._apply_masks(img))
        energy9 = self._combine_to_9(energy16)

        H, W = energy9.shape[1:]
        flat = energy9.reshape(9, -1).T
        flat_std = self.scaler.fit_transform(flat) if fit_scaler else self.scaler.transform(flat)
        self.energy9_std = flat_std.T.reshape(9, H, W)

        if mode == 'pointwise':
            feats = self.energy9_std.reshape(9, -1).T
            pos = np.array([(i, j) for i in range(H) for j in range(W)])
            return feats, pos, 1
        else:
            fs = fragment_size
            feats, pos = [], []
            for i in range(0, H - fs + 1, fs):
                for j in range(0, W - fs + 1, fs):
                    patch = self.energy9_std[:, i:i+fs, j:j+fs]
                    vec = np.median(patch.reshape(9, -1), axis=1)
                    feats.append(vec)
                    pos.append((i, j))
            return np.asarray(feats), np.asarray(pos), fs

    def _roi_feature_vectors(self, roi_box, mode, fragment_size):
        y1, x1, y2, x2 = roi_box
        if mode == 'pointwise':
            return self.energy9_std[:, y1:y2, x1:x2].reshape(9, -1).T
        else:
            fs = fragment_size
            feats = []
            for i in range(y1, y2 - fs + 1, fs):
                for j in range(x1, x2 - fs + 1, fs):
                    patch = self.energy9_std[:, i:i+fs, j:j+fs]
                    feats.append(np.median(patch.reshape(9, -1), axis=1))
            return np.asarray(feats) if len(feats) else np.empty((0, 9))

    def fit_prototypes_from_rois(self, class_rois, mode='pointwise', fragment_size=15,
                                 delta_percentile=95, robust=True):
        prototypes_per_class, deltas_per_class = [], []
        for boxes in class_rois:
            if not boxes:
                prototypes_per_class.append(np.empty((0, 9)))
                deltas_per_class.append(np.inf)
                continue
            class_protos, in_dists = [], []
            for box in boxes:
                feats = self._roi_feature_vectors(box, mode, fragment_size)
                if feats.size == 0:
                    continue
                proto = np.median(feats, axis=0) if robust else np.mean(feats, axis=0)
                class_protos.append(proto)
                d = np.mean(np.abs(feats - proto[None, :]), axis=1)
                in_dists.append(d)
            if class_protos:
                P = np.vstack(class_protos)
                prototypes_per_class.append(P)
                d_all = np.concatenate(in_dists, axis=0) if in_dists else np.array([np.inf])
                deltas_per_class.append(np.percentile(d_all, delta_percentile))
            else:
                prototypes_per_class.append(np.empty((0, 9)))
                deltas_per_class.append(np.inf)
        return prototypes_per_class, np.asarray(deltas_per_class)

    @staticmethod
    def _min_dist_per_class(feat_vecs, prototypes_per_class):
        N = feat_vecs.shape[0]
        C = len(prototypes_per_class)
        D = np.full((N, C), np.inf, dtype=np.float64)
        for c, P in enumerate(prototypes_per_class):
            if P.size == 0:
                continue
            diff = np.abs(feat_vecs[:, None, :] - P[None, :, :])  # (N,M,9)
            D[:, c] = diff.mean(axis=2).min(axis=1)
        return D

    def segment_with_prototypes(self, features, positions, scale,
                                prototypes_per_class, deltas_per_class,
                                unknown_label=-1):
        H, W = self.energy9_std.shape[1:]
        seg = (np.zeros((H, W), dtype=np.int32) + unknown_label)
        D = self._min_dist_per_class(features, prototypes_per_class)
        best_c = np.argmin(D, axis=1)
        best_d = D[np.arange(D.shape[0]), best_c]
        accept = best_d < deltas_per_class[best_c]
        for i, (y, x) in enumerate(positions):
            if accept[i]:
                if scale == 1:
                    seg[y, x] = best_c[i]
                else:
                    seg[y:y+scale, x:x+scale] = best_c[i]
        return seg

# ---------- Interactive utilities ----------
def clamp_roi_fixed_size(cx, cy, size, H, W):
    size = int(size)
    size = max(1, min(size, min(H, W)))
    y1 = int(cy - size // 2)
    x1 = int(cx - size // 2)
    y1 = max(0, min(y1, H - size))
    x1 = max(0, min(x1, W - size))
    return [y1, x1, y1 + size, x1 + size]

def draw_rois(frame_bgr, rois_per_class, current_class, colors, roi_size):
    vis = frame_bgr.copy()
    for ci, boxes in enumerate(rois_per_class):
        color = colors[ci % len(colors)]
        for (y1, x1, y2, x2) in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"C{ci}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(vis,
        f"[Class {current_class}] ROI={roi_size}px  |  1..9 switch  |  +/- size  |  Enter  |  Backspace  |  R  |  Esc",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return vis

def key_to_class_index(k):
    if ord('1') <= k <= ord('9'):
        return k - ord('1')
    persian = {0x06F1:0,0x06F2:1,0x06F3:2,0x06F4:3,0x06F5:4,0x06F6:5,0x06F7:6,0x06F8:7,0x06F9:8}
    if k in persian: return persian[k]
    arabic  = {0x0661:0,0x0662:1,0x0663:2,0x0664:3,0x0665:4,0x0666:5,0x0667:6,0x0668:7,0x0669:8}
    if k in arabic:  return arabic[k]
    return None

def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        y1, x1, y2, x2 = clamp_roi_fixed_size(cx=x, cy=y, size=state["roi_size"], H=state["H"], W=state["W"])
        state["rois_per_class"][state["current_class"]].append([y1, x1, y2, x2])

# ---------- Saving helpers ----------
def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    m, M = float(np.min(img)), float(np.max(img))
    if M - m < 1e-9:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip(255 * (img - m) / (M - m), 0, 255).astype(np.uint8)

def save_energy9_grid(energy9_std, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for k in range(9):
        ax = axes[k//3, k%3]
        ax.imshow(energy9_std[k], cmap='viridis')
        ax.set_title(f'E{k+1}', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_rois_image(gray, rois_per_class_compact, out_path):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for ci, boxes in enumerate(rois_per_class_compact):
        color = ROI_COLORS_BGR[ci % len(ROI_COLORS_BGR)]
        for (y1, x1, y2, x2) in boxes:
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rgb, f"C{ci}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, rgb)
    return rgb

def save_segmentation(seg, C, out_path, delta_pct):
    cmap = ListedColormap(np.array(PALETTE_RGB[:C]) / 255.0)
    cmap.set_bad(color='white')
    seg_masked = np.ma.masked_where(seg < 0, seg)
    fig = plt.figure(figsize=(7.5, 7))
    plt.imshow(seg_masked, cmap=cmap, vmin=0, vmax=C-1)
    plt.title(f"Segmentation (per-class δ = {delta_pct}th)")
    plt.axis('off')
    handles = [Patch(facecolor=np.array(PALETTE_RGB[i])/255.0, label=f'C{i}') for i in range(C)]
    plt.legend(handles=handles, ncol=min(C, 6), loc='lower center',
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

# ---------- Main ----------
if __name__ == "__main__":
    IMAGE_PATH    = "Im501.jpg"
    MODE          = 'fragmental'      # 'pointwise' or 'fragmental'
    FRAGMENT_SIZE = 5
    ENERGY_WIN    = 25
    NORM_WIN      = 15
    ROI_SIZE      = 70
    DELTA_PCTL    = 99

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    H, W = img.shape[:2]

    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    mode_tag = "PW" if MODE == 'pointwise' else "FRAG"
    prefix = f"{stem}_{mode_tag}"

    cv2.imwrite(f"{prefix}_original.png", ensure_uint8(img))

    if MODE == 'fragmental':
        ROI_SIZE = max(FRAGMENT_SIZE, (ROI_SIZE // FRAGMENT_SIZE) * FRAGMENT_SIZE)
    ROI_SIZE = min(ROI_SIZE, min(H, W))

    MAX_CLASSES = 9
    rois_per_class = [[] for _ in range(MAX_CLASSES)]
    state = {
        "rois_per_class": rois_per_class,
        "current_class": 0,
        "roi_size": ROI_SIZE,
        "H": H, "W": W,
        "mode": MODE,
        "fragment_size": FRAGMENT_SIZE
    }

    win = "Interactive ROI Picker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse, param=state)

    print("Controls: 1..9 select | Left-click add ROI | +/- size | Backspace undo | R reset | Enter run | Esc quit")

    while True:
        canvas = draw_rois(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                           state["rois_per_class"], state["current_class"], ROI_COLORS_BGR, state["roi_size"])
        cv2.imshow(win, canvas)
        k = cv2.waitKey(30) & 0xFFFFFFFF

        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit(0)

        cls_idx = key_to_class_index(k)
        if cls_idx is not None:
            state["current_class"] = cls_idx
            continue

        if k in (ord('+'), ord('=')):
            inc = state["fragment_size"] if state["mode"] == 'fragmental' else 2
            state["roi_size"] = min(max(5, state["roi_size"] + inc), min(H, W))
            if state["mode"] == 'fragmental':
                fs = state["fragment_size"]
                state["roi_size"] = max(fs, (state["roi_size"] // fs) * fs)
            continue
        if k in (ord('-'), ord('_')):
            dec = state["fragment_size"] if state["mode"] == 'fragmental' else 2
            state["roi_size"] = max(5, state["roi_size"] - dec)
            if state["mode"] == 'fragmental':
                fs = state["fragment_size"]
                state["roi_size"] = max(fs, (state["roi_size"] // fs) * fs)
            continue

        if k == 8:
            c = state["current_class"]
            if state["rois_per_class"][c]:
                state["rois_per_class"][c].pop()
            continue
        if k in (ord('r'), ord('R')):
            state["rois_per_class"] = [[] for _ in range(MAX_CLASSES)]
            continue

        if k in (13, 10):
            cv2.destroyAllWindows()
            valid = [boxes for boxes in state["rois_per_class"] if len(boxes) > 0]
            if not valid:
                print("No ROIs selected. Exiting.")
                raise SystemExit(0)

            segm = LawsTextureSegmentation(energy_window_size=ENERGY_WIN, norm_window_size=NORM_WIN)
            features, positions, scale = segm.extract_features(
                img, mode=state["mode"], fragment_size=state["fragment_size"], fit_scaler=True
            )

            save_energy9_grid(segm.energy9_std, f"{prefix}_energy9_grid.png")
            _ = save_rois_image(img, valid, f"{prefix}_rois.png")

            prototypes_per_class, deltas_per_class = segm.fit_prototypes_from_rois(
                valid, mode=state["mode"], fragment_size=state["fragment_size"],
                delta_percentile=DELTA_PCTL, robust=True
            )
            seg = segm.segment_with_prototypes(features, positions, scale,
                                               prototypes_per_class, deltas_per_class)

            C = len(valid)
            save_segmentation(seg, C, f"{prefix}_segmentation.png", DELTA_PCTL)

            cmap = ListedColormap(np.array(PALETTE_RGB[:C]) / 255.0)
            cmap.set_bad(color='white')
            seg_masked = np.ma.masked_where(seg < 0, seg)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(cv2.cvtColor(cv2.imread(f"{prefix}_rois.png"), cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original + ROIs'); axes[0].axis('off')
            axes[1].imshow(seg_masked, cmap=cmap, vmin=0, vmax=C-1)
            axes[1].set_title(f'Segmentation (per-class δ = {DELTA_PCTL}th)'); axes[1].axis('off')
            handles = [Patch(facecolor=np.array(PALETTE_RGB[i])/255.0, label=f'C{i}') for i in range(C)]
            fig.legend(handles=handles, ncol=min(C, 6), loc='lower center',
                       bbox_to_anchor=(0.5, 0.0), frameon=False)
            plt.tight_layout(); plt.show()

            known = int(np.sum(seg >= 0)); total = int(seg.size)
            print(f"Known pixels: {known}/{total} ({100.0*known/total:.2f}%)")
            break
