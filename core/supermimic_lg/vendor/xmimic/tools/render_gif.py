"""Render the release GIF: the G1 driven by an XMimic policy (a tools/rollout_cpu.py recording) next to the headline result.

    python tools/rollout_cpu.py <ckpt>.npz --out rollout.npz      # defaults: dance1_subject2 from 60 s, 12 s
    python tools/render_gif.py rollout.npz                        # seconds 1-11 of it -> figures/xmimic.gif

figures/xmimic.gif was made this way from the checkpoint at iteration 2000 (54 s of training) of the first release's 4-GPU
run with seed 0. The numbers on the GIF come from results/ (tools/make_figures.py), not from arguments. Rendering needs
OpenGL (MUJOCO_GL=glfw by default; egl on a headless machine).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_figures import load_results, summarize  # noqa: E402

W, H, PANEL = 960, 540, 470  # GIF size and the width of the robot panel on the right
FPS = 20
BG = (255, 255, 255)  # the palette of the paper and the release video: white, ink, XMimic teal, BlinkMimic gray
INK, TEAL, GRAY = (29, 29, 31), (0, 176, 190), (110, 110, 115)


def font(size, bold=False):
    """Helvetica Neue where available (macOS), else matplotlib's DejaVu Sans."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=1 if bold else 0)
    except OSError:
        import matplotlib

        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name), size)


def scene_xml():
    """assets/g1/scene.xml in the release video's look: a white floor that only shows the robot's soft shadow (lit enough
    to render pure white) under a white sky, with soft key and fill lights."""
    path = os.path.join(ROOT, "assets", "g1", "scene.xml")
    xml = open(path).read()
    xml = xml.replace('meshdir="assets"', f'meshdir="{os.path.join(ROOT, "assets", "g1", "assets")}"')
    xml = re.sub(r"<light[^>]*/>", "", xml)
    xml = xml.replace('rgba="0.7 0.7 0.7 1"', 'rgba="0.66 0.67 0.70 1"').replace('rgba="0.2 0.2 0.2 1"', 'rgba="0.15 0.15 0.17 1"')
    xml = re.sub(r'<texture type="2d"[^>]*name="groundplane"[^>]*/>',
                 '<texture type="2d" name="groundplane" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/>', xml)
    xml = re.sub(r'<material name="groundplane"[^>]*/>', '<material name="groundplane" texture="groundplane" texuniform="true" '
                 'texrepeat="1 1" reflectance="0" emission="0.22" specular="0" shininess="0"/>', xml)  # fmt: skip
    xml = re.sub(r"<texture[^>]*type=\"skybox\"[^>]*/>", "", xml)
    lights = ('<light name="key" pos="1.2 -1.6 5" dir="-0.18 0.24 -1" directional="true" diffuse="0.52 0.52 0.52" '
              'specular="0.18 0.18 0.18" castshadow="true"/><light name="fill" pos="-2 2 4" dir="0.4 -0.4 -1" directional="true" '
              'diffuse="0.27 0.27 0.28" specular="0 0 0" castshadow="false"/>')  # fmt: skip
    xml = xml.replace("</worldbody>", lights + "</worldbody>", 1)
    xml = re.sub(r"<visual>.*?</visual>", "", xml, flags=re.S)
    vis = ('<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="8192" offsamples="8"/>'
           '<headlight ambient="0.40 0.40 0.40" diffuse="0.1 0.1 0.1" specular="0.04 0.04 0.04"/>'
           '<map fogstart="4" fogend="9" haze="0" shadowclip="2.5" shadowscale="0.8"/><rgba fog="1 1 1 1"/></visual>')  # fmt: skip
    sky = '<texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/>'
    return xml.replace("<asset>", vis + "<asset>" + sky, 1)


def nlerp(q0, q1, w):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - w) * q0 + w * q1
    return q / np.linalg.norm(q)


def robot_frames(rec, start, seconds, fq):
    """The robot panel at FPS frames per second, interpolated from the 50 Hz recording; the camera follows the root."""
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, PANEL)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    cam = mujoco.MjvCamera()
    qpos = rec["qpos"].astype(np.float64)
    T = len(qpos)
    k = 40  # smoothed camera target: moving average of the root over +-0.8 s
    xy = np.pad(qpos[:, fq : fq + 2], ((k, k), (0, 0)), mode="edge")
    look = np.stack([np.convolve(xy[:, j], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for j in range(2)], 1)
    out = []
    for i in range(int(round(seconds * FPS))):
        s = min((start + i / FPS) * 50, T - 1.001)
        i0, w = int(s), s - int(s)
        q = (1 - w) * qpos[i0] + w * qpos[i0 + 1]
        q[fq + 3 : fq + 7] = nlerp(qpos[i0, fq + 3 : fq + 7], qpos[i0 + 1, fq + 3 : fq + 7], w)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        cam.azimuth, cam.elevation, cam.distance = 150 + 2.0 * i / FPS, -6, 2.45
        cam.lookat[:] = [*look[i0], 0.74]
        renderer.update_scene(data, camera=cam)
        out.append(Image.fromarray(renderer.render()))
    return out


def text_panel(summ):
    """The left panel: title, the speedup, and the two times behind it (all from results/)."""
    img = Image.new("RGB", (W - PANEL, H), BG)
    d = ImageDraw.Draw(img)
    x = 40
    held_x, held_b = np.mean(summ["xmimic_awlp"]["held"]), np.mean(summ["blinkmimic"]["held"])
    d.text((x, 46), "XMimic", font=font(48, bold=True), fill=INK)
    d.text((x - 6, 104), f"{held_b / held_x:.1f}×", font=font(156, bold=True), fill=TEAL)
    d.text((x, 278), "faster training to", font=font(34), fill=INK)
    d.text((x, 319), "BlinkMimic's target", font=font(34), fill=INK)
    num = font(44, bold=True)
    off = int(max(d.textlength(f"{v:.2f} min", font=num) for v in (held_x, held_b))) + 16
    labels = ("XMimic, 4 H100s", "BlinkMimic, 1 H100")
    size = 31  # the largest label size that fits next to the numbers
    while max(d.textlength(t, font=font(size)) for t in labels) > img.width - x - off - 14:
        size -= 1
    for y, v, col, t in ((392, held_x, TEAL, labels[0]), (452, held_b, GRAY, labels[1])):
        d.text((x, y), f"{v:.2f} min", font=num, fill=col)
        d.text((x + off, y + 44 - size - 2), t, font=font(size), fill=INK)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout", help="tools/rollout_cpu.py output")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "xmimic.gif"))
    ap.add_argument("--start", type=float, default=1.0, help="start in the recording (s)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fade", type=int, default=6, help="frames of fade at the loop boundary")
    args = ap.parse_args()
    rec = np.load(args.rollout)
    from xmimic.model import load_meta

    fq = load_meta()["free_qpos_adr"]
    left = text_panel(summarize(load_results()))
    robots = robot_frames(rec, args.start, args.seconds, fq)
    frames = []
    for i, rob in enumerate(robots):
        a = min(1.0, (i + 1) / args.fade, (len(robots) - i) / args.fade)  # fade the robot in and out at the loop boundary
        rob = Image.blend(Image.new("RGB", rob.size, BG), rob, a) if a < 1 else rob
        f = Image.new("RGB", (W, H), BG)
        f.paste(left, (0, 0))
        f.paste(rob, (W - PANEL, 0))
        frames.append(f)
    # one palette for all frames (no flicker), from a montage of samples
    sample = Image.new("RGB", (W, H * 4))
    for j, k in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        sample.paste(frames[k], (0, H * j))
    pal = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=False, disposal=1)
    print(f"{len(q)} frames {W}x{H} at {FPS} fps -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
