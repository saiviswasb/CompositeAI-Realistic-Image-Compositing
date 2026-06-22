"""
CompositeAI — Image Compositing & Shadow Harmonization Pipeline
Two-path extraction:
  Path A — KNN Matting (Classical): interactive trimap with auto-prefill
  Path B — Deep Learning: rembg, BRIA, RVM, MODNet, ViTMatte

Pipeline: Extraction → Placement → Harmonization → Shadow → A/B Testing

Install:
    pip install gradio pillow numpy scipy opencv-python rembg onnxruntime
    pip install scikit-image torch torchvision transformers diffusers accelerate pymatting
"""

import json, traceback, io, base64
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import gradio as gr


# ══════════════════════════════════════════════════════════════════
# PREREQUISITE CHECKER
# ══════════════════════════════════════════════════════════════════

import importlib, os, sys, subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def check_prerequisites():
    """
    Check ONLY mandatory items that do NOT auto-download.
    Returns list of (name, status, message, fix_cmd) tuples.
    """
    results = []

    def check(label, fn, fix=""):
        try:
            msg = fn()
            results.append((label, True, msg or "Ready", fix))
        except Exception as e:
            results.append((label, False, str(e)[:90], fix))

    # ── Mandatory Python packages ─────────────────────────────
    mandatory_pkgs = [
        ("gradio",      "gradio",      "pip install gradio"),
        ("numpy",       "numpy",       "pip install numpy"),
        ("opencv",      "cv2",         "pip install opencv-python"),
        ("Pillow",      "PIL",         "pip install Pillow"),
        ("scipy",       "scipy",       "pip install scipy"),
        ("scikit-image","skimage",     "pip install scikit-image"),
        ("torch",       "torch",       "pip install torch torchvision"),
        ("transformers","transformers","pip install transformers"),
        ("rembg",       "rembg",       "pip install rembg onnxruntime"),
        ("pymatting",   "pymatting",   "pip install pymatting"),
        ("diffusers",   "diffusers",   "pip install diffusers accelerate"),
    ]
    for name, mod, fix in mandatory_pkgs:
        def _chk(m=mod):
            importlib.import_module(m)
            mod_obj = importlib.import_module(m)
            v = getattr(mod_obj, '__version__', None)
            return f"v{v}" if v else "installed"
        check(f"{name}", _chk, fix)

    # ── libcom (must be manually installed) ──────────────────
    def chk_libcom():
        from libcom import ImageHarmonizationModel, ShadowGenerationModel
        return "installed"
    check("libcom",
          chk_libcom,
          "git clone https://github.com/bcmi/libcom.git && cd libcom && pip install -e .")

    # ── GPSDiffusion weights — NOT auto-download ─────────────
    # These come from ModelScope and must be manually copied
    def chk_shadow_weights():
        try:
            import libcom
            lib_path  = os.path.dirname(libcom.__file__)
            shad_path = os.path.join(lib_path, "shadow_generation", "pretrained_models")
            if not os.path.exists(shad_path):
                raise Exception(f"Folder missing: pretrained_models/")
            files = []
            total = 0
            for root, dirs, fs in os.walk(shad_path):
                for f in fs:
                    fp = os.path.join(root, f)
                    files.append(f)
                    try: total += os.path.getsize(fp)
                    except: pass
            if not files:
                raise Exception("Empty folder — copy weights from ModelScope (~6GB)")
            gb = total / (1024**3)
            if gb < 1.0:
                raise Exception(f"Only {gb:.1f}GB found — expected ~6GB (ControlNet + IP-Adapter)")
            return f"{len(files)} files — {gb:.1f}GB"
        except ImportError:
            raise Exception("libcom not installed — install it first")
    check("GPSDiffusion weights (ControlNet + IP-Adapter ~6GB)",
          chk_shadow_weights,
          "Copy from: ~/libcom/libcom/shadow_generation/pretrained_models/")

    return results


def build_prereq_html(results):
    """
    Top-right slide-in notification.
    Appears 2s after Gradio loads via js= parameter trigger.
    Auto-dismisses 5s if all pass. Stays if any fail.
    """
    all_ok   = all(ok for _, ok, _, _ in results)
    ok_count = sum(1 for _, ok, _, _ in results)
    total    = len(results)

    rows = ""
    for name, ok, msg, fix in results:
        icon  = "✅" if ok else "❌"
        color = "#22C55E" if ok else "#EF4444"
        bg    = "#0c1a10" if ok else "#1a0c0c"

        # Build copy-able command block for failed items
        if not ok and fix:
            fix_html = f"""
                <div style="margin-top:6px;background:#080b10;border-radius:6px;
                            border:1px solid #C1121F44;overflow:hidden;">
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:4px 8px;background:#C1121F18;border-bottom:1px solid #C1121F33;">
                        <span style="color:#C1121F;font-size:9px;font-weight:800;
                                     text-transform:uppercase;letter-spacing:.1em;">
                            Run this command
                        </span>
                        <button onclick="navigator.clipboard.writeText('{fix}').then(function(){{
                                    this.textContent='Copied!';
                                    setTimeout(function(){{document.querySelectorAll('.copy-btn').forEach(function(b){{b.textContent='Copy'}})}},1500);
                                }}.bind(this))"
                                class="copy-btn"
                                style="background:#C1121F;color:#fff;border:none;
                                       border-radius:4px;padding:2px 8px;font-size:9px;
                                       font-weight:700;cursor:pointer;font-family:Outfit,sans-serif;">
                            Copy
                        </button>
                    </div>
                    <div style="padding:6px 8px;font-family:monospace;font-size:10px;
                                color:#E5C07B;line-height:1.6;word-break:break-all;">
                        {fix}
                    </div>
                </div>"""
        else:
            fix_html = ""

        rows += f"""
        <div style="display:flex;align-items:flex-start;gap:9px;
                    padding:8px 10px;background:{bg};
                    border-radius:8px;margin-bottom:5px;
                    border-left:3px solid {color};">
            <span style="font-size:13px;margin-top:1px;flex-shrink:0;">{icon}</span>
            <div style="min-width:0;flex:1;">
                <div style="color:#FFFFFF;font-weight:700;font-size:12px;
                            line-height:1.3;">{name}</div>
                <div style="color:#666;font-size:10px;margin-top:1px;">{msg}</div>
                {fix_html}
            </div>
        </div>"""

    accent     = "#22C55E" if all_ok else "#C1121F"
    title_icon = "✅" if all_ok else "⚠️"
    title_text = f"All {total} checks passed" if all_ok else f"{total-ok_count} item(s) need attention"
    countdown  = 5 if all_ok else 0

    all_fixes = [fix for _, ok, _, fix in results if not ok and fix]
    install_block = ""
    if all_fixes:
        cmds = "<br>".join(all_fixes)
        install_block = f"""
        <div style="margin-top:10px;padding:9px 10px;background:#080b10;
                    border-radius:8px;border:1px solid #2A2F3A;">
            <div style="color:#888;font-size:9px;font-weight:800;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;">
                Fix commands
            </div>
            <code style="color:#E5383B;font-size:9px;line-height:1.8;display:block;
                         word-break:break-all;">{cmds}</code>
        </div>"""

    html = f"""
<div id="prereq-notif" data-countdown="{countdown}" style="
    position:fixed;top:20px;right:20px;
    width:370px;max-height:82vh;
    background:linear-gradient(160deg,#1C2130 0%,#131820 100%);
    border:1.5px solid #2E3748;
    border-top:3px solid {accent};
    border-radius:16px;
    box-shadow:0 20px 60px rgba(0,0,0,0.8),0 0 0 1px rgba(255,255,255,0.04);
    z-index:99999;
    font-family:Outfit,system-ui,sans-serif;
    opacity:0;
    transform:translateY(-24px) scale(0.97);
    transition:opacity .4s ease,transform .45s cubic-bezier(.34,1.4,.64,1);
    pointer-events:none;
    overflow:hidden;
">
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:13px 14px 11px;border-bottom:1px solid #252D3A;">
        <div style="display:flex;align-items:center;gap:9px;">
            <span style="font-size:16px;">{title_icon}</span>
            <div>
                <div style="color:#555E6E;font-size:9px;font-weight:800;
                            letter-spacing:.14em;text-transform:uppercase;">
                    CompositeAI · Prerequisites
                </div>
                <div style="color:#FFFFFF;font-size:13px;font-weight:800;margin-top:1px;">
                    {title_text}
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:{accent};font-size:11px;font-weight:800;
                         background:rgba(0,0,0,0.35);padding:3px 8px;
                         border-radius:6px;border:1px solid {accent}55;">
                {ok_count}/{total}
            </span>
            <button id="prereq-x"
                    style="background:#1C2130;border:1px solid #2E3748;color:#666;
                           width:22px;height:22px;border-radius:6px;cursor:pointer;
                           font-size:12px;line-height:1;padding:0;flex-shrink:0;">
                ✕
            </button>
        </div>
    </div>

    <div style="padding:10px 10px 4px;overflow-y:auto;
                max-height:calc(82vh - 110px);">
        {rows}
        {install_block}
    </div>

    <div style="padding:8px 12px 11px;border-top:1px solid #252D3A;
                display:flex;align-items:center;gap:8px;">
        <div id="prereq-msg" style="flex:1;color:#444;font-size:10px;font-weight:600;">
            {'Closing in 5s...' if all_ok else 'Restart after fixing'}
        </div>
        <button id="prereq-ok"
                style="padding:5px 14px;
                       background:{'linear-gradient(135deg,#22C55E,#16a34a)' if all_ok else 'linear-gradient(135deg,#C1121F,#E5383B)'};
                       color:#fff;border:none;border-radius:7px;
                       font-weight:700;font-size:11px;cursor:pointer;
                       font-family:Outfit,sans-serif;">
            {'Got it ✓' if all_ok else 'Dismiss'}
        </button>
    </div>
</div>

"""

    return html


# ── Run checks at startup ─────────────────────────────────────────
print("\n[CompositeAI] Checking prerequisites...")
_PREREQ_RESULTS = check_prerequisites()
ok_n   = sum(1 for _,s,_,_ in _PREREQ_RESULTS if s)
fail_n = sum(1 for _,s,_,_ in _PREREQ_RESULTS if not s)
if fail_n:
    print(f"[CompositeAI] ⚠️  {fail_n} prerequisite(s) missing:")
    for name,ok,msg,fix in _PREREQ_RESULTS:
        if not ok:
            print(f"  ❌ {name}: {msg}")
            if fix: print(f"     → {fix}")
else:
    print(f"[CompositeAI] ✅ All {ok_n} checks passed")
_PREREQ_HTML = build_prereq_html(_PREREQ_RESULTS)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def np2pil(arr): return Image.fromarray(arr.astype(np.uint8))
def pil2np(img): return np.array(img)
def pil_to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
def b64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64.split(",",1)[-1])))
def get_device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"
def np_to_b64(arr): return pil_to_b64(np2pil(arr))


# ══════════════════════════════════════════════════════════════════
# PATH A — KNN MATTING
# ══════════════════════════════════════════════════════════════════

def auto_prefill_trimap(fg_np):
    if fg_np is None: return None, None, None, "Upload an image first."
    try:
        from rembg import remove
        img   = np2pil(fg_np).convert("RGB")
        rgba  = remove(img)
        alpha = np.array(rgba.split()[3])
        trimap = np.full_like(alpha, 128, dtype=np.uint8)
        trimap[alpha > 200] = 255
        trimap[alpha < 50]  = 0
        kernel  = np.ones((15,15), np.uint8)
        fg_dil  = cv2.dilate((trimap==255).astype(np.uint8), kernel, iterations=2)
        bg_dil  = cv2.dilate((trimap==0).astype(np.uint8),  kernel, iterations=2)
        overlap = (fg_dil & bg_dil).astype(bool)
        trimap[fg_dil==0]=0; trimap[bg_dil==0]=255; trimap[overlap]=128
        h,w  = trimap.shape
        vis  = np.zeros((h,w,3), dtype=np.uint8)
        vis[trimap==255]=[0,200,0]; vis[trimap==0]=[200,0,0]; vis[trimap==128]=[200,200,0]
        blend = (np.array(img)*0.5 + vis*0.5).astype(np.uint8)
        return trimap, np_to_b64(blend), pil_to_b64(img), "✅ Auto-prefill done! Refine yellow edges."
    except Exception as e:
        traceback.print_exc(); return None,None,None,f"Error: {e}"

def run_knn_matting(fg_np, trimap_np):
    if fg_np is None or trimap_np is None: return None,None,"Run auto-prefill first."
    try:
        from pymatting import estimate_alpha_knn
        img_rgb  = np2pil(fg_np).convert("RGB")
        alpha    = estimate_alpha_knn(np.array(img_rgb).astype(np.float64)/255.0,
                                       trimap_np.astype(np.float64)/255.0)
        alpha_np = np.clip(alpha*255,0,255).astype(np.uint8)
        rgba     = img_rgb.convert("RGBA"); rgba.putalpha(Image.fromarray(alpha_np))
        b64 = pil_to_b64(rgba)
        return pil2np(rgba), b64, "✅ KNN matting complete!"
    except Exception as e:
        traceback.print_exc(); return None,None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# PATH B — DEEP LEARNING EXTRACTION
# ══════════════════════════════════════════════════════════════════

def dl_rembg_general(img):
    from rembg import remove; return remove(img.convert("RGB"))
def dl_rembg_human(img):
    from rembg import remove, new_session
    return remove(img.convert("RGB"), session=new_session("u2net_human_seg"))
def dl_rembg_silueta(img):
    from rembg import remove, new_session
    return remove(img.convert("RGB"), session=new_session("silueta"))
def dl_bria(img):
    import torch
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("image-segmentation", model="briaai/RMBG-1.4",
                        trust_remote_code=True, device=0 if torch.cuda.is_available() else -1)
    result = pipe(img.convert("RGB"))
    item = result[0] if isinstance(result,list) else result
    mask = item['mask'] if isinstance(item,dict) else item
    if not isinstance(mask,Image.Image): mask=Image.fromarray(np.array(mask).astype(np.uint8))
    rgba=img.convert("RGBA"); rgba.putalpha(mask.convert("L")); return rgba
def dl_rvm(img):
    import torch; from torchvision import transforms
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=torch.hub.load("PeterL1n/RobustVideoMatting","mobilenetv3",pretrained=True,trust_repo=True).to(device).eval()
    inp=transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(device)
    rec=[None]*4
    with torch.no_grad(): fgr,pha,*rec=model(inp,*rec,downsample_ratio=1)
    alpha_np=(pha.squeeze().cpu().numpy()*255).astype(np.uint8)
    w,h=img.size; alpha_np=cv2.resize(alpha_np,(w,h),interpolation=cv2.INTER_LANCZOS4)
    rgba=img.convert("RGB").convert("RGBA"); rgba.putalpha(Image.fromarray(alpha_np)); return rgba
def dl_modnet(img):
    import torch; from torchvision import transforms
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=torch.hub.load("ZHKKKe/MODNet","modnet_photographic_portrait_matting",pretrained=True,trust_repo=True).to(device).eval()
    w,h=img.size; nw,nh=(w//32)*32,(h//32)*32
    inp=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])(
        img.convert("RGB").resize((nw,nh),Image.LANCZOS)).unsqueeze(0).to(device)
    with torch.no_grad(): _,_,matte=model(inp,True)
    alpha_np=(matte.squeeze().cpu().numpy()*255).astype(np.uint8)
    alpha_np=cv2.resize(alpha_np,(w,h),interpolation=cv2.INTER_LANCZOS4)
    rgba=img.convert("RGB").convert("RGBA"); rgba.putalpha(Image.fromarray(alpha_np)); return rgba
def dl_vitmatte(img, rough_alpha=None):
    import torch; from transformers import VitMatteForImageMatting,VitMatteImageProcessor
    device="cuda" if torch.cuda.is_available() else "cpu"
    processor=VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
    model=VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(device).eval()
    img_rgb=img.convert("RGB"); w,h=img_rgb.size
    if rough_alpha is None:
        from rembg import remove; rough_alpha=np.array(remove(img_rgb).split()[3])
    trimap=np.full((h,w),128,dtype=np.uint8); trimap[rough_alpha>200]=255; trimap[rough_alpha<50]=0
    kernel=np.ones((15,15),np.uint8)
    fg_d=cv2.dilate((trimap==255).astype(np.uint8),kernel,iterations=2)
    bg_d=cv2.dilate((trimap==0).astype(np.uint8),kernel,iterations=2)
    trimap[fg_d==0]=0; trimap[bg_d==0]=255; trimap[(fg_d&bg_d).astype(bool)]=128
    inputs=processor(images=img_rgb,trimaps=Image.fromarray(trimap),return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad(): alpha=model(**inputs).alphas
    alpha_np=(alpha.squeeze().cpu().numpy()*255).astype(np.uint8)
    alpha_np=cv2.resize(alpha_np,(w,h),interpolation=cv2.INTER_LANCZOS4)
    rgba=img_rgb.convert("RGBA"); rgba.putalpha(Image.fromarray(alpha_np)); return rgba

DL_METHODS = {
    "U2Net — General objects":          dl_rembg_general,
    "U2Net Human — People & portraits": dl_rembg_human,
    "Silueta — Fine edges & hair":      dl_rembg_silueta,
    "BRIA RMBG-1.4 — Best quality":     dl_bria,
    "RobustVideoMatting — Best hair":   dl_rvm,
    "MODNet — Portrait matting":        dl_modnet,
    "ViTMatte — Best overall quality":  dl_vitmatte,
}

def do_dl_extract(fg_np, method_name):
    if fg_np is None: return None,None,"Upload a foreground image first."
    try:
        img  = np2pil(fg_np).convert("RGB")
        rgba = DL_METHODS.get(method_name, dl_rembg_human)(img)
        b64  = pil_to_b64(rgba)
        return pil2np(rgba), b64, f"✅ Extracted [{method_name}]"
    except Exception as e:
        traceback.print_exc(); return None,None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# PLACEMENT
# ══════════════════════════════════════════════════════════════════

def do_composite(bg_np, fg_b64, x_pct, y_pct, scale_pct):
    if bg_np is None:  return None,None,"Upload a background image."
    if not fg_b64:     return None,None,"Complete extraction first."
    try:
        bg=np2pil(bg_np).convert("RGB"); fg=b64_to_pil(fg_b64).convert("RGBA")
        bw,bh=bg.size
        fw=max(10,int(bw*scale_pct/100.0)); fh=max(10,int(fw*fg.height/fg.width))
        fg_r=fg.resize((fw,fh),Image.LANCZOS)
        px=max(0,min(bw-fw,int(x_pct/100.0*bw-fw/2)))
        py=max(0,min(bh-fh,int(y_pct/100.0*bh-fh/2)))
        comp=bg.copy(); comp.paste(fg_r,(px,py),fg_r)
        pos=json.dumps(dict(px=px,py=py,fw=fw,fh=fh,bw=bw,bh=bh))
        return pil2np(comp),pos,f"✅ Placed at ({px},{py}), {fw}×{fh}"
    except Exception as e:
        traceback.print_exc(); return None,None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# HARMONIZATION
# ══════════════════════════════════════════════════════════════════

def harmonize_histogram(comp,fg_b64,pos,intensity):
    from skimage.exposure import match_histograms
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
    fg_rgb=pil2np(fg)[:,:,:3]; alpha=pil2np(fg)[:,:,3:4]/255.0
    bg_reg=comp[max(0,py-20):min(bh,py+fh+20),max(0,px-20):min(bw,px+fw+20),:3]
    if bg_reg.size==0: bg_reg=comp[:,:,:3]
    matched=match_histograms(fg_rgb,bg_reg,channel_axis=-1).astype(np.uint8)
    blended=(matched*intensity+fg_rgb*(1-intensity)).astype(np.uint8)
    result=comp.copy(); y2,x2=min(bh,py+fh),min(bw,px+fw)
    result[py:y2,px:x2,:3]=(blended[:y2-py,:x2-px]*alpha[:y2-py,:x2-px]+result[py:y2,px:x2,:3]*(1-alpha[:y2-py,:x2-px])).astype(np.uint8)
    return result

def harmonize_reinhard(comp,fg_b64,pos,intensity):
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
    fg_rgb=pil2np(fg)[:,:,:3].astype(np.float32); alpha=pil2np(fg)[:,:,3:4]/255.0
    bg_reg=comp[max(0,py-40):min(bh,py+fh+40),max(0,px-40):min(bw,px+fw+40),:3].astype(np.float32)
    if bg_reg.size==0: bg_reg=comp[:,:,:3].astype(np.float32)
    result=comp.copy()
    for c in range(3):
        sm,ss=fg_rgb[:,:,c].mean(),fg_rgb[:,:,c].std()+1e-6
        tm,ts=bg_reg[:,:,c].mean(),bg_reg[:,:,c].std()+1e-6
        b=np.clip((fg_rgb[:,:,c]-sm)*(ts/ss)+tm,0,255)*intensity+fg_rgb[:,:,c]*(1-intensity)
        y2,x2=min(bh,py+fh),min(bw,px+fw)
        result[py:y2,px:x2,c]=(b[:y2-py,:x2-px]*alpha[:y2-py,:x2-px,0]+result[py:y2,px:x2,c]*(1-alpha[:y2-py,:x2-px,0])).astype(np.uint8)
    return result

def harmonize_lab(comp,fg_b64,pos,intensity):
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
    fg_rgb=pil2np(fg)[:,:,:3]; alpha=pil2np(fg)[:,:,3:4]/255.0
    fg_lab=cv2.cvtColor(fg_rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_reg=comp[max(0,py-40):min(bh,py+fh+40),max(0,px-40):min(bw,px+fw+40),:3]
    if bg_reg.size==0: bg_reg=comp[:,:,:3]
    bg_lab=cv2.cvtColor(bg_reg,cv2.COLOR_RGB2LAB).astype(np.float32)
    result=comp.copy()
    for c in range(3):
        sm,ss=fg_lab[:,:,c].mean(),fg_lab[:,:,c].std()+1e-6
        tm,ts=bg_lab[:,:,c].mean(),bg_lab[:,:,c].std()+1e-6
        fg_lab[:,:,c]=np.clip((fg_lab[:,:,c]-sm)*(ts/ss)+tm,0 if c==0 else -128,100 if c==0 else 127)*intensity+fg_lab[:,:,c]*(1-intensity)
    harmonized=cv2.cvtColor(fg_lab.astype(np.uint8),cv2.COLOR_LAB2RGB)
    y2,x2=min(bh,py+fh),min(bw,px+fw)
    result[py:y2,px:x2,:3]=(harmonized[:y2-py,:x2-px]*alpha[:y2-py,:x2-px]+result[py:y2,px:x2,:3]*(1-alpha[:y2-py,:x2-px])).astype(np.uint8)
    return result

def harmonize_wb(comp,fg_b64,pos,intensity):
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
    fg_rgb=pil2np(fg)[:,:,:3].astype(np.float32); alpha=pil2np(fg)[:,:,3:4]/255.0
    bg_m=comp[:,:,:3].astype(np.float32).mean(axis=(0,1)); fg_m=fg_rgb.mean(axis=(0,1))+1e-6
    adj=np.clip(fg_rgb*(1+(bg_m/fg_m-1)*intensity),0,255).astype(np.uint8)
    result=comp.copy(); y2,x2=min(bh,py+fh),min(bw,px+fw)
    result[py:y2,px:x2,:3]=(adj[:y2-py,:x2-px]*alpha[:y2-py,:x2-px]+result[py:y2,px:x2,:3]*(1-alpha[:y2-py,:x2-px])).astype(np.uint8)
    return result

def harmonize_idih(comp,fg_b64,pos,intensity):
    from libcom import ImageHarmonizationModel
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    model=ImageHarmonizationModel(device=get_device(),model_type='PCTNet')
    mask_np=np.zeros((bh,bw),dtype=np.uint8); mask_np[py:min(bh,py+fh),px:min(bw,px+fw)]=255
    out=model(np2pil(comp).convert("RGB"),Image.fromarray(mask_np))
    out_np=pil2np(out.convert("RGB")) if isinstance(out,Image.Image) else out[:,:,:3]
    return (out_np.astype(np.float32)*intensity+comp.astype(np.float32)*(1-intensity)).astype(np.uint8)

def harmonize_rainnet(comp,fg_b64,pos,intensity):
    from libcom import ImageHarmonizationModel
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    model=ImageHarmonizationModel(device=get_device(),model_type='LBM')
    mask_np=np.zeros((bh,bw),dtype=np.uint8); mask_np[py:min(bh,py+fh),px:min(bw,px+fw)]=255
    out=model(np2pil(comp).convert("RGB"),Image.fromarray(mask_np))
    out_np=pil2np(out.convert("RGB")) if isinstance(out,Image.Image) else out[:,:,:3]
    return (out_np.astype(np.float32)*intensity+comp.astype(np.float32)*(1-intensity)).astype(np.uint8)

def harmonize_hf(comp,fg_b64,pos,intensity):
    try:
        import torch; from transformers import pipeline as hf_pipeline
        p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
        pipe=hf_pipeline("depth-estimation",model="LiheYoung/depth-anything-small-hf",
                          device=0 if torch.cuda.is_available() else -1)
        depth=pipe(np2pil(comp).convert("RGB"))['depth']
        depth_np=pil2np(depth.convert("L")).astype(np.float32)/255.0
        fg_region=comp[py:min(bh,py+fh),px:min(bw,px+fw),:3].astype(np.float32)
        surround=np.concatenate([depth_np[max(0,py-20):py,px:min(bw,px+fw)].flatten(),
                                  depth_np[min(bh,py+fh):min(bh,py+fh+20),px:min(bw,px+fw)].flatten()])
        if surround.size>0:
            lum=np.clip(surround.mean()/(fg_region.mean()/255.0+1e-6),0.5,2.0)
            fg_region=np.clip(fg_region*(1+(lum-1)*intensity),0,255)
        fg_b=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
        alpha=pil2np(fg_b)[:,:,3:4]/255.0; result=comp.copy(); y2,x2=min(bh,py+fh),min(bw,px+fw)
        result[py:y2,px:x2,:3]=(fg_region[:y2-py,:x2-px].astype(np.uint8)*alpha[:y2-py,:x2-px]+result[py:y2,px:x2,:3]*(1-alpha[:y2-py,:x2-px])).astype(np.uint8)
        return result
    except Exception as e:
        print(f"[HF] {e}"); return harmonize_lab(comp,fg_b64,pos,intensity)

HARMONIZE_METHODS = {
    "Histogram Matching":                        harmonize_histogram,
    "Reinhard Color Transfer":                   harmonize_reinhard,
    "LAB Color Transfer":                        harmonize_lab,
    "White Balance":                             harmonize_wb,
    "Deep Learning — PCTNet (libcom)":           harmonize_idih,
    "Deep Learning — LBM (libcom)":              harmonize_rainnet,
    "Deep Learning — Depth-Aware (HuggingFace)": harmonize_hf,
}

def do_harmonize(comp_np,fg_b64,pos,method_name,intensity):
    if comp_np is None: return None,None,"Build composite first."
    if not pos:         return None,None,"Build composite first."
    try:
        result=HARMONIZE_METHODS.get(method_name,harmonize_lab)(comp_np.copy(),fg_b64,pos,float(intensity))
        return result,result,f"✅ Harmonized [{method_name}] at {int(intensity*100)}%"
    except Exception as e:
        traceback.print_exc(); return None,None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# SHADOW SYNTHESIS
# ══════════════════════════════════════════════════════════════════

def shadow_geometric(input_np,fg_b64,pos,bg_np,opacity,length,blur):
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    bg=np2pil(bg_np).convert("RGB"); fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)
    gray=np.array(bg.convert("L"),dtype=np.float32); h,w=gray.shape
    sm=gaussian_filter(gray,sigma=max(h,w)*0.05)
    ys,xs=np.where(sm>=np.percentile(sm,95))
    lx,ly=(np.mean(xs)/w,np.mean(ys)/h) if len(xs)>0 else (0.3,0.2)
    ar=np.arctan2(0.5-ly,0.5-lx); ad=np.degrees(ar)
    elev=max(15.0,min(75.0,90.0-np.sqrt((lx-.5)**2+(ly-.5)**2)*120)); er=np.radians(max(5.0,elev))
    alpha=np.array(fg.split()[3],dtype=np.float32)/255.0
    sil=(alpha>0.5).astype(np.float32); sh,sw=sil.shape
    sl=min(int(sh*length/np.tan(er)),sh); sdx,sdy=int(sl*np.cos(ar)),int(sl*np.sin(ar))
    M=np.float32([[1,-np.cos(ar)/np.tan(er),max(0,sdx)],[0,1,max(0,sdy)]])
    sfg=cv2.warpAffine(sil,M,(sw,sh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    rows=np.where(np.any(alpha>0.5,axis=1))[0]
    if len(rows)>0:
        br=rows[-1]; cols=np.where(alpha[br]>0.5)[0]
        if len(cols)>0:
            cx=int(np.mean(cols)); ew=max(10,(cols[-1]-cols[0])//2); eh=max(4,ew//4)
            cont=np.zeros((sh,sw),dtype=np.float32)
            cv2.ellipse(cont,(cx,min(br+eh//2,sh-1)),(ew,eh),0,0,360,0.5,-1)
            sfg=np.clip(sfg+gaussian_filter(cont,sigma=eh*1.5),0,1)
    bl=gaussian_filter(sfg,sigma=int(blur))
    dt=cv2.distanceTransform((sfg<0.5).astype(np.uint8),cv2.DIST_L2,5)
    sfg=np.clip(bl*(1-np.clip(dt/(int(blur)*3),0,1)*0.6),0,1)
    sf=np.zeros((bh,bw),dtype=np.float32)
    sx,sy=max(0,px+sdx),max(0,py+sdy)
    x1,y1=max(0,sx),max(0,sy); x2,y2=min(bw,sx+sw),min(bh,sy+sh)
    if x2>x1 and y2>y1:
        sf[y1:y2,x1:x2]=np.maximum(sf[y1:y2,x1:x2],sfg[y1-sy:y1-sy+(y2-y1),x1-sx:x1-sx+(x2-x1)])
    ba=np.array(bg,dtype=np.float32)
    sc=(int(ba[bh//2:,:,0].mean()*.25),int(ba[bh//2:,:,1].mean()*.25),int(ba[bh//2:,:,2].mean()*.30))
    bgr=np.array(bg.convert("RGBA"),dtype=np.float32); sm2=sf*opacity
    for c,s in enumerate(sc): bgr[:,:,c]=bgr[:,:,c]*(1-sm2)+s*sm2
    bgr[:,:,3]=255; res=Image.fromarray(bgr.astype(np.uint8),"RGBA"); res.paste(fg,(px,py),fg)
    return pil2np(res.convert("RGB")),f"✅ Geometric | angle {ad:.0f}° elev {elev:.0f}°"

_shadow_model=None
def get_shadow_model():
    global _shadow_model
    if _shadow_model is None:
        from libcom import ShadowGenerationModel
        device=get_device(); print(f"[GPSDiffusion] Loading on {device}...")
        _shadow_model=ShadowGenerationModel(device=device); print("[GPSDiffusion] Ready.")
    return _shadow_model

def shadow_gpsdiffusion(input_np,fg_b64,pos,bg_np,opacity,length,blur):
    p=json.loads(pos); px,py,fw,fh,bw,bh=p['px'],p['py'],p['fw'],p['fh'],p['bw'],p['bh']
    bg=np2pil(bg_np).convert("RGB"); fg=b64_to_pil(fg_b64).convert("RGBA").resize((fw,fh),Image.LANCZOS)

    # Build full object mask
    alpha=np.array(fg.split()[3]); mask=np.zeros((bh,bw),dtype=np.uint8)
    x2,y2=min(bw,px+fw),min(bh,py+fh); mask[py:y2,px:x2]=alpha[:y2-py,:x2-px]
    mask=(mask>128).astype(np.uint8)*255

    bright=np.array(bg,dtype=np.float32).mean()/255.0
    intens="strong" if bright>0.65 else "medium" if bright>0.4 else "soft"
    comp=input_np[:,:,:3].astype(np.uint8)

    # ── Optimisation 1: Crop around object ──────────────────────
    margin   = 100
    cx1      = max(0, px - margin)
    cy1      = max(0, py - margin)
    cx2      = min(bw, px + fw + margin)
    cy2      = min(bh, py + fh + margin)
    crop_comp = comp[cy1:cy2, cx1:cx2]
    crop_mask = mask[cy1:cy2, cx1:cx2]

    # ── Optimisation 2: Resize crop to 512px ────────────────────
    h_crop, w_crop = crop_comp.shape[:2]
    scale    = min(1.0, 512 / max(h_crop, w_crop))
    small_comp = cv2.resize(crop_comp,
                             (int(w_crop*scale), int(h_crop*scale)),
                             interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(crop_mask,
                             (small_comp.shape[1], small_comp.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # ── Optimisation 3: Generate only ONE shadow ─────────────────
    model   = get_shadow_model()
    print(f"[GPSDiffusion] Running on crop {small_comp.shape} (orig {bw}x{bh})")
    results = model(shadowfree_img=small_comp, object_mask=small_mask, number=1)
    if not results: raise RuntimeError("No results from GPSDiffusion.")

    # ── Resize result back to crop size ─────────────────────────
    result = results[0]
    result = pil2np(result) if isinstance(result, Image.Image) else result
    result = result[:,:,:3]
    result = cv2.resize(result, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)

    # ── Paste crop back into full image ─────────────────────────
    final = comp.copy()
    final[cy1:cy2, cx1:cx2] = result

    # ── Fix warm shadow tint ─────────────────────────────────────
    diff = comp.astype(np.float32) - final.astype(np.float32)
    sa   = cv2.dilate((diff.mean(axis=2)>8).astype(np.uint8),
                       np.ones((3,3),np.uint8)).astype(np.float32)
    c2   = final.astype(np.float32)
    c2[:,:,0] *= (1 - sa*0.15)
    c2[:,:,2] *= (1 + sa*0.08)
    final = np.clip(c2, 0, 255).astype(np.uint8)

    if intens=="soft":
        final = (comp*0.4 + final*0.6).astype(np.uint8)

    return final, f"✅ GPSDiffusion | {intens} | brightness {bright:.2f} | crop {w_crop}x{h_crop}→512px"

SHADOW_METHODS = {
    "Geometric — instant, CPU-friendly":         shadow_geometric,
    "GPSDiffusion — deep learning, scene-aware": shadow_gpsdiffusion,
}

def do_shadow(harmonized_np,comp_np,fg_b64,pos,bg_np,method_name,opacity,length,blur):
    input_np=harmonized_np if harmonized_np is not None else comp_np
    if input_np is None: return None,"Build composite first."
    if not fg_b64:       return None,"Extract foreground first."
    if not pos:          return None,"Preview composite first."
    try:
        return SHADOW_METHODS.get(method_name,shadow_geometric)(input_np,fg_b64,pos,bg_np,opacity,float(length),float(blur))
    except Exception as e:
        traceback.print_exc(); return None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# A/B TESTING
# ══════════════════════════════════════════════════════════════════

def do_ab(comp_np,fg_b64,pos,bg_np,ha,ia,sa,oa,hb,ib,sb,ob):
    if comp_np is None or not pos: return None,None,"Complete Steps 1-3 first."
    try:
        ra=HARMONIZE_METHODS.get(ha,harmonize_lab)(comp_np.copy(),fg_b64,pos,float(ia))
        ra,_=SHADOW_METHODS.get(sa,shadow_geometric)(ra,fg_b64,pos,bg_np,oa,1.2,20)
        rb=HARMONIZE_METHODS.get(hb,harmonize_lab)(comp_np.copy(),fg_b64,pos,float(ib))
        rb,_=SHADOW_METHODS.get(sb,shadow_geometric)(rb,fg_b64,pos,bg_np,ob,1.2,20)
        return ra,rb,"✅ A/B comparison ready"
    except Exception as e:
        traceback.print_exc(); return None,None,f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# STEP 06 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════

def compute_niqe(img_np):
    """
    NIQE — Natural Image Quality Evaluator (no reference).
    Lower = more natural/realistic.
    Approximated via local variance statistics.
    """
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float64)
        # Local mean and variance (NIQE-style patch statistics)
        kernel = np.ones((7,7), np.float64) / 49
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_sq   = cv2.filter2D(gray**2, -1, kernel)
        local_var  = np.maximum(local_sq - local_mean**2, 0)
        local_std  = np.sqrt(local_var)
        # MSCN (Mean Subtracted Contrast Normalized) coefficients
        mscn = (gray - local_mean) / (local_std + 1.0)
        # Fit GGD — use kurtosis as naturalness proxy
        from scipy.stats import kurtosis
        niqe_score = float(np.abs(kurtosis(mscn.flatten())))
        return round(niqe_score, 3)
    except Exception as e:
        print(f"[NIQE] {e}"); return None

def compute_brisque(img_np):
    """
    BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator.
    Lower = better quality. Uses MSCN coefficient statistics.
    """
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float64)
        kernel = np.ones((7,7), np.float64) / 49
        mu    = cv2.filter2D(gray, -1, kernel)
        mu_sq = cv2.filter2D(gray**2, -1, kernel)
        sigma = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        mscn  = (gray - mu) / (sigma + 1.0)
        # GGD shape parameter proxy
        from scipy.stats import kurtosis, skew
        kurt = float(kurtosis(mscn.flatten()))
        skw  = float(skew(mscn.flatten()))
        # Paired product (horizontal)
        h_prod = mscn[:, :-1] * mscn[:, 1:]
        h_kurt = float(kurtosis(h_prod.flatten()))
        brisque = round(abs(kurt) + abs(skw) + abs(h_kurt), 3)
        return brisque
    except Exception as e:
        print(f"[BRISQUE] {e}"); return None

def compute_clip_score(img_np, prompt="a realistic photo of an object in a natural scene"):
    """
    CLIP Score — perceptual realism measured by CLIP image-text alignment.
    Higher = more realistic.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        proc   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        pil    = np2pil(img_np).convert("RGB")
        inputs = proc(text=[prompt], images=pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        score = float(out.logits_per_image.squeeze().cpu())
        return round(score, 3)
    except Exception as e:
        print(f"[CLIP] {e}"); return None

def compute_delta_e(img_a, img_b, mask_pos=None):
    """
    ΔE (CIEDE2000) — perceptual color difference between two images.
    Lower = better color harmonization.
    Computed in the foreground/composited region.
    """
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
        lab_a = rgb2lab(img_a.astype(np.float32) / 255.0)
        lab_b = rgb2lab(img_b.astype(np.float32) / 255.0)
        delta = deltaE_ciede2000(lab_a, lab_b)
        if mask_pos is not None:
            px,py,fw,fh,bw,bh = mask_pos
            region = delta[py:min(bh,py+fh), px:min(bw,px+fw)]
            return round(float(region.mean()), 3) if region.size > 0 else round(float(delta.mean()), 3)
        return round(float(delta.mean()), 3)
    except Exception as e:
        print(f"[DeltaE] {e}"); return None

def compute_shadow_geometry(result_np, composite_np):
    """
    Shadow Geometry Analysis — measures shadow consistency.
    Returns: shadow area %, shadow darkness, edge sharpness.
    """
    try:
        diff = composite_np.astype(np.float32) - result_np.astype(np.float32)
        diff_gray = diff.mean(axis=2)
        # Shadow mask: pixels significantly darker in result
        shadow_mask = (diff_gray > 10).astype(np.uint8)
        shadow_pct  = round(float(shadow_mask.mean() * 100), 2)
        # Average darkening in shadow region
        shadow_dark = round(float(diff_gray[shadow_mask > 0].mean()), 2) if shadow_mask.sum() > 0 else 0.0
        # Edge sharpness: Laplacian variance in shadow boundary
        kernel   = np.ones((3,3), np.uint8)
        dilated  = cv2.dilate(shadow_mask, kernel, iterations=2)
        eroded   = cv2.erode(shadow_mask, kernel, iterations=2)
        boundary = (dilated - eroded).astype(np.uint8)
        if boundary.sum() > 0:
            result_gray = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)
            lap    = cv2.Laplacian(result_gray, cv2.CV_64F)
            sharp  = round(float(np.abs(lap[boundary > 0]).mean()), 2)
        else:
            sharp = 0.0
        return shadow_pct, shadow_dark, sharp
    except Exception as e:
        print(f"[ShadowGeo] {e}"); return None, None, None

def run_evaluation(result_a, result_b, comp_np, pos):
    """Run all evaluation metrics for both A and B results."""
    if result_a is None or result_b is None:
        return "Run A/B Test first."
    try:
        pos_dict = json.loads(pos) if pos else None
        mask_pos = (pos_dict['px'],pos_dict['py'],pos_dict['fw'],
                    pos_dict['fh'],pos_dict['bw'],pos_dict['bh']) if pos_dict else None

        # NIQE
        niqe_a = compute_niqe(result_a)
        niqe_b = compute_niqe(result_b)
        # BRISQUE
        brq_a  = compute_brisque(result_a)
        brq_b  = compute_brisque(result_b)
        # CLIP
        clip_a = compute_clip_score(result_a)
        clip_b = compute_clip_score(result_b)
        # Delta E vs composite (reference)
        de_a   = compute_delta_e(result_a, comp_np, mask_pos)
        de_b   = compute_delta_e(result_b, comp_np, mask_pos)
        # Shadow geometry
        sp_a, sd_a, ss_a = compute_shadow_geometry(result_a, comp_np)
        sp_b, sd_b, ss_b = compute_shadow_geometry(result_b, comp_np)

        def fmt(v): return str(v) if v is not None else "N/A"
        def win(a, b, lower_better=True):
            if a is None or b is None: return ""
            return " ✅" if (a < b if lower_better else a > b) else " ✅"

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║              EVALUATION RESULTS                     ║",
            "╠══════════════════════════════════════════════════════╣",
            "║  METRIC                  METHOD A      METHOD B     ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  NIQE (↓ better)        {fmt(niqe_a):<13} {fmt(niqe_b):<13}║",
            f"║  BRISQUE (↓ better)     {fmt(brq_a):<13} {fmt(brq_b):<13}║",
            f"║  CLIP Score (↑ better)  {fmt(clip_a):<13} {fmt(clip_b):<13}║",
            f"║  ΔE Color (↓ better)    {fmt(de_a):<13} {fmt(de_b):<13}║",
            f"║  Shadow Area %          {fmt(sp_a):<13} {fmt(sp_b):<13}║",
            f"║  Shadow Darkness        {fmt(sd_a):<13} {fmt(sd_b):<13}║",
            f"║  Shadow Sharpness       {fmt(ss_a):<13} {fmt(ss_b):<13}║",
            "╚══════════════════════════════════════════════════════╝",
        ]

        # Winner summary
        scores = {"A": 0, "B": 0}
        metrics = [
            (niqe_a,  niqe_b,  True),
            (brq_a,   brq_b,   True),
            (clip_a,  clip_b,  False),
            (de_a,    de_b,    True),
        ]
        for a, b, lower in metrics:
            if a is not None and b is not None:
                if (a < b if lower else a > b): scores["A"] += 1
                else: scores["B"] += 1

        winner = "A" if scores["A"] > scores["B"] else "B" if scores["B"] > scores["A"] else "TIE"
        lines.append(f"\n🏆 Winner: Method {winner}  (A: {scores['A']} pts | B: {scores['B']} pts)")

        return "\n".join(lines)

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"



def adjust_shadow_darkness(shadow_np, comp_np, darkness, pos=None):
    """
    Adjusts ONLY the shadow pixels generated by the diffusion model.

    Method:
    1. Diff = composite - shadow_result  →  finds exactly where shadow was added
    2. Threshold diff > 5 to isolate shadow region precisely
    3. Restrict to ground region below object (using pos if available)
    4. Smooth mask edges with Gaussian blur
    5. Re-blend ONLY shadow pixels: orig + (shadow-orig) * darkness
       → darkness=1.0: original full shadow
       → darkness=0.5: shadow is half as dark
       → darkness=0.0: shadow completely removed, orig restored
    Non-shadow pixels are mathematically untouched (multiplied by 1.0).
    """
    if shadow_np is None or comp_np is None:
        return None, "Generate shadow first."
    try:
        shadow = shadow_np[:,:,:3].astype(np.float32)
        orig   = comp_np[:,:,:3].astype(np.float32)
        h, w   = orig.shape[:2]

        if shadow.shape[:2] != orig.shape[:2]:
            shadow = cv2.resize(shadow, (w, h))

        # ── Step 1: Find shadow region ──────────────────────────
        # Shadow pixels are where diffusion made pixels DARKER
        diff      = orig - shadow          # positive = darkened
        diff_gray = diff.mean(axis=2)      # per-pixel mean darkening

        # Adaptive threshold: use percentile to handle varying shadow strength
        thresh = max(5.0, float(np.percentile(diff_gray[diff_gray > 0], 30))
                     if (diff_gray > 0).sum() > 100 else 5.0)
        raw_mask = (diff_gray > thresh).astype(np.float32)

        # ── Step 2: Restrict to ground region below FG object ───
        if pos:
            try:
                p  = json.loads(pos)
                py = p['py']; fh = p['fh']; bh = p['bh']
                # Shadow only appears below the object
                ground_mask = np.zeros((h, w), dtype=np.float32)
                ground_start = max(0, py + fh - 10)  # small overlap for contact shadow
                ground_mask[ground_start:bh, :] = 1.0
                raw_mask = raw_mask * ground_mask
            except:
                pass  # fallback: use full image mask

        # ── Step 3: Morphological cleanup ───────────────────────
        # Remove small noise, keep connected shadow region
        kernel   = np.ones((5,5), np.uint8)
        raw_mask_u8 = (raw_mask * 255).astype(np.uint8)
        cleaned  = cv2.morphologyEx(raw_mask_u8, cv2.MORPH_CLOSE, kernel)
        cleaned  = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                                     np.ones((3,3), np.uint8))

        # ── Step 4: Smooth edges ─────────────────────────────────
        shadow_mask = cv2.GaussianBlur(cleaned.astype(np.float32)/255.0,
                                        (21, 21), 0)
        shadow_mask = np.clip(shadow_mask, 0, 1)[:,:,np.newaxis]

        # ── Step 5: Apply darkness ONLY to shadow region ─────────
        # Formula: result = orig + (shadow - orig) * darkness
        #   darkness=1.0 → result = shadow  (full diffusion output)
        #   darkness=0.0 → result = orig    (shadow completely gone)
        shadow_diff = shadow - orig          # how much diffusion darkened
        result = orig + shadow_diff * darkness * shadow_mask
        result = np.clip(result, 0, 255).astype(np.uint8)

        shadow_coverage = round(float((shadow_mask > 0.1).mean() * 100), 1)
        return result, (f"✅ Shadow-only edit | Darkness: {int(darkness*100)}% | "
                        f"Shadow coverage: {shadow_coverage}% of image")
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"


def move_shadow(shadow_np, comp_np, shift_x, shift_y, darkness=1.0):
    """
    Move ONLY the shadow region by shift_x, shift_y pixels.
    shift_x: positive = right, negative = left
    shift_y: positive = down,  negative = up
    darkness: shadow intensity (0=none, 1=full)
    """
    if shadow_np is None or comp_np is None:
        return None, "Generate shadow first."
    try:
        shadow = shadow_np[:,:,:3].astype(np.float32)
        orig   = comp_np[:,:,:3].astype(np.float32)
        h, w   = orig.shape[:2]

        if shadow.shape[:2] != orig.shape[:2]:
            shadow = cv2.resize(shadow, (w, h))

        # ── Detect shadow mask ──────────────────────────────────
        diff      = orig - shadow
        diff_gray = diff.mean(axis=2)
        thresh    = max(5.0, float(np.percentile(diff_gray[diff_gray > 0], 30))
                        if (diff_gray > 0).sum() > 100 else 5.0)
        raw_mask  = (diff_gray > thresh).astype(np.float32)

        # Smooth mask
        raw_mask = cv2.GaussianBlur(raw_mask, (15, 15), 0)
        raw_mask = np.clip(raw_mask, 0, 1)

        # ── Shift the shadow mask ───────────────────────────────
        sx, sy   = int(shift_x), int(shift_y)
        M        = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted_mask = cv2.warpAffine(raw_mask, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

        # ── Also shift the shadow content itself ────────────────
        # Take just the darkening from original shadow
        shadow_darkening = diff  # how much the shadow darkened each pixel
        shifted_dark     = cv2.warpAffine(shadow_darkening, M, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)

        # ── Recomposite: orig - shifted_darkening * mask ────────
        mask3    = shifted_mask[:,:,np.newaxis]
        result   = orig - shifted_dark * mask3 * darkness
        result   = np.clip(result, 0, 255).astype(np.uint8)

        moved_px = round(float((shifted_mask > 0.1).mean() * 100), 1)
        return result, (f"✅ Shadow moved ({sx:+d}px, {sy:+d}px) | "
                        f"Darkness: {int(darkness*100)}% | "
                        f"Coverage: {moved_px}%")
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {e}"


# ══════════════════════════════════════════════════════════════════
# EMBEDDED PIPELINE JS
# ══════════════════════════════════════════════════════════════════

PIPELINE_JS = """
function initPipelineAnimation() {
    var map = [
        ["section-extraction",    "card-extract"],
        ["section-placement",     "card-placement"],
        ["section-harmonization", "card-harmonization"],
        ["section-shadow",        "card-shadow"],
        ["section-ab",            "card-ab"]
    ];

    var cards = document.querySelectorAll(".pipeline-card");
    if (!cards.length) {
        setTimeout(initPipelineAnimation, 1000);
        return;
    }

    // Staggered fade-in for all cards
    cards.forEach(function(card, index) {
        setTimeout(function() {
            card.classList.add("visible");
        }, index * 250);
    });

    // Find the scrollable container — Gradio puts content in a scrollable div
    // Try multiple candidates
    function getScrollRoot() {
        var candidates = [
            document.querySelector('.gradio-container'),
            document.querySelector('.main'),
            document.querySelector('.contain'),
            document.documentElement,
            document.body
        ];
        for (var i = 0; i < candidates.length; i++) {
            var el = candidates[i];
            if (el && el.scrollHeight > el.clientHeight) return el;
        }
        return window;
    }

    function getTop(el) {
        var rect = el.getBoundingClientRect();
        return rect.top;
    }

    function getSectionOrder() {
        var result = [];
        map.forEach(function(item) {
            var section = document.getElementById(item[0]);
            if (section) {
                result.push({ section: section, cardId: item[1] });
            }
        });
        return result;
    }

    function updateActiveCard() {
        var sections = getSectionOrder();
        if (!sections.length) return;

        var viewportMid = window.innerHeight * 0.45;
        var active = null;

        // Find the section closest to middle of viewport from above
        for (var i = 0; i < sections.length; i++) {
            var top = getTop(sections[i].section);
            if (top <= viewportMid) {
                active = sections[i];
            }
        }

        // If nothing is above midpoint yet, use first
        if (!active) active = sections[0];

        // Update active card
        document.querySelectorAll(".pipeline-card").forEach(function(c) {
            c.classList.remove("active");
        });
        var card = document.getElementById(active.cardId);
        if (card) card.classList.add("active");
    }

    // Listen on both window and all scroll containers
    window.addEventListener("scroll", updateActiveCard, true);
    document.addEventListener("scroll", updateActiveCard, true);

    // Also poll every 300ms as fallback for Gradio's internal scroll
    setInterval(updateActiveCard, 300);

    // Initial call
    setTimeout(updateActiveCard, 500);

    console.log("[CompositeAI] Pipeline animation loaded");
}

// Run reliably
window.addEventListener("load", function() {
    setTimeout(initPipelineAnimation, 2000);
});

// Gradio re-render protection
new MutationObserver(function() {
    var cards   = document.querySelectorAll(".pipeline-card");
    var visible = document.querySelectorAll(".pipeline-card.visible");
    if (cards.length && !visible.length) {
        initPipelineAnimation();
    }
}).observe(document.body, { childList: true, subtree: true });

// ── Prereq notifier — runs from js= parameter ──────────────────
setTimeout(function() {
    var el = document.getElementById('prereq-notif');
    if (!el) { console.log('[CompositeAI] prereq-notif not found yet'); return; }
    el.style.opacity       = '1';
    el.style.transform     = 'translateY(0) scale(1)';
    el.style.pointerEvents = 'auto';

    var xb  = document.getElementById('prereq-x');
    var okb = document.getElementById('prereq-ok');
    var msg = document.getElementById('prereq-msg');

    function hide() {
        el.style.opacity       = '0';
        el.style.transform     = 'translateY(-24px) scale(0.97)';
        el.style.pointerEvents = 'none';
    }
    if (xb)  xb.onclick  = hide;
    if (okb) okb.onclick = hide;

    // Auto-dismiss countdown (only if data-countdown set)
    var cd = parseInt(el.getAttribute('data-countdown') || '0');
    if (cd > 0) {
        var timer = setInterval(function() {
            cd--;
            if (msg) msg.textContent = 'Closing in ' + cd + 's...';
            if (cd <= 0) { clearInterval(timer); hide(); }
        }, 1000);
    }
    console.log('[CompositeAI] Prereq notifier shown');
}, 2500);
"""

# ══════════════════════════════════════════════════════════════════
# THEME CSS
# ══════════════════════════════════════════════════════════════════

DARK_CSS = """html, html::before, html::after,
body, body::before, body::after {
    background: #0B0D12 !important;
    background-color: #0B0D12 !important;
}
* { transition: background 0s !important; }
gradio-app, gradio-app * { background-color: #0B0D12 !important; }
.generating { background: #0B0D12 !important; }
.pending { background: #11161F !important; }

@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

/* ===== FULL VIEWPORT DARK ===== */
html, body {
    background: #0B0D12 !important;
    min-height: 100vh !important;
    width: 100% !important;
    margin: 0 !important; padding: 0 !important;
    font-family: 'Outfit', system-ui, sans-serif !important;
}
html *, body * {
    background-color: #0B0D12;
    font-family: 'Outfit', system-ui, sans-serif;
    color: #FFFFFF;
}
#root, gradio-app, .app, .main, .contain, footer {
    background: #0B0D12 !important;
}

/* ===== CONTAINER — left panel with room for sidebar ===== */
.gradio-container {
    background: #0B0D12 !important;
    width: 62vw !important;
    max-width: 62vw !important;
    min-width: 480px !important;
    margin-left: 2vw !important;
    margin-right: 36vw !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 24px 20px !important;
    float: none !important;
}

/* ===== PRESENTATION SIDEBAR ===== */
.pipeline-sidebar {
    position: fixed;
    top: 0; right: 0;
    width: 34vw;
    height: 100vh;
    overflow-y: auto;
    padding: 40px 28px;
    background: linear-gradient(180deg, rgba(20,25,34,.98), rgba(11,13,18,.98));
    border-left: 1px solid #3B4352;
    z-index: 999;
}
.pipeline-title {
    color: #E5383B;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 25px;
    letter-spacing: -0.02em;
}
.pipeline-card {
    opacity: 0;
    transform: translateY(50px);
    transition: opacity 0.8s ease, transform 0.8s ease, box-shadow 0.4s ease;
    margin-bottom: 18px;
    background: linear-gradient(180deg, #1A1F28, #141922);
    border: 1px solid #3B4352;
    border-left: 5px solid #C1121F;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 28px rgba(0,0,0,.5);
}
.pipeline-card.visible {
    opacity: 1 !important;
    transform: translateY(0) !important;
    box-shadow: 0 0 22px rgba(193,18,31,.22), 0 10px 30px rgba(0,0,0,.55);
}
.pipeline-card.active {
    border-left: 5px solid #E5383B !important;
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 0 0 35px rgba(193,18,31,.4), 0 12px 32px rgba(0,0,0,.65) !important;
}
.pipeline-card:not(.active) {
    border-left: 5px solid transparent !important;
}
.pipeline-card:not(.active) .pipeline-heading,
.pipeline-card:not(.active) .pipeline-text,
.pipeline-card:not(.active) .pipeline-stage {
    color: #555B66 !important;
}
.pipeline-card.active .pipeline-heading {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}
.pipeline-card.active .pipeline-text {
    color: #D8DCE5 !important;
    font-weight: 600 !important;
}
.pipeline-card.active .pipeline-stage {
    color: #E5383B !important;
    font-weight: 800 !important;
}
.pipeline-stage {
    color: #E5383B;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-bottom: 6px;
    background: transparent !important;
}
.pipeline-heading {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 10px;
    background: transparent !important;
}
.pipeline-text {
    color: #D8DCE5;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.75;
    background: transparent !important;
}

/* ===== ALL TEXT LEFT ALIGNED ===== */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    text-align: left !important;
    background: transparent !important;
}
p, label, span, li, td, th, a {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    text-align: left !important;
    background: transparent !important;
}

/* ===== CARDS ===== */
.gr-group, .gr-box, .gr-form, .gr-panel, .block, .gap {
    background: linear-gradient(180deg, #1A1F28 0%, #141922 100%) !important;
    border: 1.5px solid #3B4352 !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5) !important;
    transition: border-color .2s ease;
}
.gr-group:hover, .block:hover {
    border-color: #C1121F !important;
    box-shadow: 0 0 0 1px rgba(193,18,31,0.25), 0 10px 30px rgba(0,0,0,0.6) !important;
}

/* ===== INPUTS ===== */
input:not([type="range"]):not([type="radio"]):not([type="checkbox"]), textarea {
    background: #11161F !important;
    color: #FFFFFF !important;
    border: 1px solid #3B4352 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
input:focus, textarea:focus {
    border-color: #C1121F !important;
    box-shadow: 0 0 0 2px rgba(193,18,31,0.2) !important;
    outline: none !important;
}
::placeholder { color: #6B7280 !important; }

/* ===== DROPDOWNS ===== */
select {
    background: #0B0D12 !important;
    color: #FFFFFF !important;
    border: 1px solid #3B4352 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
}
select option { background: #0B0D12 !important; color: #FFFFFF !important; }
select:focus { border-color: #C1121F !important; outline: none !important; }
ul[role="listbox"], li[role="option"] {
    background: #0B0D12 !important; color: #FFFFFF !important; font-weight: 600 !important;
}
li[role="option"]:hover { background: #1A1F28 !important; }
li[role="option"][aria-selected="true"] { background: #C1121F !important; }

/* ===== RADIO ===== */
fieldset { background: transparent !important; border: none !important; }
fieldset label {
    background: #11161F !important;
    color: #FFFFFF !important;
    border: 1px solid #3B4352 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    cursor: pointer !important;
    font-weight: 700 !important;
    transition: all 0.15s !important;
}
fieldset label:has(input:checked) {
    background: #C1121F !important;
    border-color: #E5383B !important;
    color: #FFFFFF !important;
}

/* ===== BUTTONS ===== */
button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    border: none !important;
    letter-spacing: 0.04em !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
}
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #C1121F, #E5383B) !important;
    color: #FFFFFF !important;
}
button.primary:hover, button[variant="primary"]:hover {
    filter: brightness(1.12) !important;
    box-shadow: 0 0 22px rgba(193,18,31,0.5) !important;
    transform: translateY(-1px) !important;
}
button.secondary, button[variant="secondary"] {
    background: #1A1F28 !important;
    color: #FFFFFF !important;
    border: 1px solid #3B4352 !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    border-color: #C1121F !important;
}

/* ===== SLIDERS — shorter, left aligned ===== */
input[type="range"] {
    accent-color: #C1121F !important;
    width: 70% !important;
    display: block !important;
    margin: 4px 0 !important;
}

/* ===== IMAGE BOXES ===== */
.image-container, .gr-image, [data-testid="image"] {
    border: 1.5px solid #3B4352 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
.image-frame, .label-wrap, .label-wrap *,
.upload-container, .upload-container *,
[data-testid="image"] > div,
.image-preview, .image-preview * {
    background: #11161F !important;
    color: #FFFFFF !important;
    border-color: #3B4352 !important;
}
.icon-button, .toolbar, .toolbar * {
    background: #141922 !important;
    color: #FFFFFF !important;
    border-color: #3B4352 !important;
}

/* ===== SELECTION / SCROLLBAR ===== */
::selection { background: rgba(193,18,31,0.3) !important; color: #FFFFFF !important; }
::-moz-selection { background: rgba(193,18,31,0.3) !important; color: #FFFFFF !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0B0D12; }
::-webkit-scrollbar-thumb { background: #C1121F; border-radius: 2px; }

/* ===== TABLES ===== */
table { border-collapse: collapse; width: 100%; }
th { background: #141922 !important; color: #C1121F !important; font-size: 11px;
     text-transform: uppercase; padding: 10px; font-weight: 800; }
td { background: #0f1318 !important; color: #FFFFFF !important; padding: 8px 10px;
     border-bottom: 1px solid #1e2530; }
tr:hover td { background: #141922 !important; }

/* ===== ACCORDION ===== */
details, .accordion {
    background: #141922 !important;
    border: 1px solid #3B4352 !important;
    border-radius: 10px !important;
}

/* ===== ROW / COL — left aligned ===== */
.gr-row, .row { justify-content: flex-start !important; }
.gr-column, .col { align-items: flex-start !important; }
"""


# ══════════════════════════════════════════════════════════════════
# GRADIO UI — Two Column Layout
# ══════════════════════════════════════════════════════════════════

with gr.Blocks(title="CompositeAI", js=PIPELINE_JS) as demo:

    fg_b64_state     = gr.State(None)
    trimap_state     = gr.State(None)
    pos_state        = gr.State(None)
    harmonized_state = gr.State(None)

    gr.HTML(f"""
    <style>
        html, body, gradio-app, #root, .app, .main, .contain {{
            background: #0B0D12 !important;
            background-color: #0B0D12 !important;
        }}
        {DARK_CSS}
    </style>
    <script>
        (function(){{
            var s = document.createElement('style');
            s.textContent = 'html,body,gradio-app{{background:#0B0D12!important;background-color:#0B0D12!important}}';
            document.head.insertBefore(s, document.head.firstChild);
            document.documentElement.style.cssText = 'background:#0B0D12!important';
            if(document.body) document.body.style.cssText = 'background:#0B0D12!important';
        }})();
    </script>
    """)

    # ── Prerequisite popup ──────────────────────────────────────
    gr.HTML(_PREREQ_HTML)

    # Header
    gr.Markdown("# CompositeAI")
    gr.Markdown("Image Matting · Color Harmonization · Shadow Synthesis")

    gr.HTML("""
<div class="pipeline-sidebar">
  <div class="pipeline-title">CompositeAI Pipeline</div>

  <div class="pipeline-card" id="card-extract">
    <div class="pipeline-stage">Stage 01</div>
    <div class="pipeline-heading">Foreground Extraction</div>
    <div class="pipeline-text">
      Extract the target object using classical KNN Matting
      with a user-refined trimap, or modern deep-learning
      segmentation networks. Fine structures like hair,
      clothing boundaries, and transparent regions are preserved.
    </div>
  </div>

  <div class="pipeline-card" id="card-placement">
    <div class="pipeline-stage">Stage 02</div>
    <div class="pipeline-heading">Object Placement</div>
    <div class="pipeline-text">
      Position the extracted foreground naturally inside
      a new scene while maintaining realistic scale,
      location, and aspect ratio consistency.
    </div>
  </div>

  <div class="pipeline-card" id="card-harmonization">
    <div class="pipeline-stage">Stage 03</div>
    <div class="pipeline-heading">Color Harmonization</div>
    <div class="pipeline-text">
      Adapt foreground appearance to the target environment
      using classical color transfer methods and deep-learning
      harmonization models (iDIH, RainNet, Depth-Aware).
    </div>
  </div>

  <div class="pipeline-card" id="card-shadow">
    <div class="pipeline-stage">Stage 04</div>
    <div class="pipeline-heading">Shadow Synthesis</div>
    <div class="pipeline-text">
      Generate physically plausible contact and cast shadows
      using geometric projection or scene-aware GPSDiffusion
      (CVPR 2025) deep learning models.
    </div>
  </div>

  <div class="pipeline-card" id="card-ab">
    <div class="pipeline-stage">Stage 05</div>
    <div class="pipeline-heading">A/B Evaluation</div>
    <div class="pipeline-text">
      Compare multiple harmonization and shadow generation
      strategies side by side to identify the most visually
      convincing composite result.
    </div>
  </div>

  <div class="pipeline-card" id="card-eval">
    <div class="pipeline-stage">Stage 06</div>
    <div class="pipeline-heading">Quantitative Metrics</div>
    <div class="pipeline-text">
      NIQE · BRISQUE · CLIP Score · ΔE Color · Shadow Geometry.
      Automatic scoring of both methods to identify the
      objectively superior composite.
    </div>
  </div>
</div>
""")


    gr.HTML('<div id="section-extraction"></div>')
    gr.Markdown("### 01 — EXTRACTION")
    fg_upload = gr.Image(label="Upload Foreground Image", type="numpy")
    extract_mode = gr.Radio(
        choices=["KNN Matting (Classical — interactive trimap)",
                 "Deep Learning (auto removal)"],
        value="Deep Learning (auto removal)",
        label="Extraction Method",
    )

    with gr.Group(visible=True) as dl_group:
        gr.Markdown("#### Deep Learning")
        dl_method = gr.Dropdown(choices=list(DL_METHODS.keys()),
                                 value="U2Net Human — People & portraits", label="DL Method")
        dl_btn    = gr.Button("Extract Background", variant="primary")
        dl_out    = gr.Image(label="Extracted Result", type="numpy", interactive=False)
        dl_status = gr.Textbox(label="", interactive=False, show_label=False)

    with gr.Group(visible=False) as knn_group:
        gr.Markdown("#### KNN Matting — Interactive Trimap")
        gr.Markdown("1. Auto-Prefill → 2. Refine yellow edges → 3. Run KNN")
        prefill_btn    = gr.Button("🤖 Auto-Prefill Trimap", variant="primary")
        prefill_status = gr.Textbox(label="", interactive=False, show_label=False)
        trimap_opacity = gr.Slider(0.1, 1.0, value=0.7, step=0.05,
                                    label="Trimap Opacity (lower = see original through)")
        trimap_editor  = gr.ImageEditor(label="Trimap Editor — Green=FG | Red=BG | Yellow=Unknown",
                                         type="numpy", height=500)
        knn_btn    = gr.Button("Run KNN Matting", variant="primary")
        knn_out    = gr.Image(label="KNN Result", type="numpy", interactive=False)
        knn_status = gr.Textbox(label="", interactive=False, show_label=False)

    # Step 2: Placement
    gr.HTML('<div id="section-placement"></div>')
    gr.Markdown("### 02 — PLACEMENT")
    bg_input = gr.Image(label="Upload Background Scene", type="numpy")
    with gr.Row():
        x_sl = gr.Slider(0,100,value=50,step=1,label="X (%)")
        y_sl = gr.Slider(0,100,value=70,step=1,label="Y (%)")
        s_sl = gr.Slider(5,80,value=30,step=1,label="Scale (%)")
    comp_btn     = gr.Button("Preview Composite", variant="secondary")
    comp_preview = gr.Image(label="Composite Preview", type="numpy", interactive=False)
    comp_status  = gr.Textbox(label="", interactive=False, show_label=False)

    # Step 3: Harmonization
    gr.HTML('<div id="section-harmonization"></div>')
    gr.Markdown("### 03 — HARMONIZATION")
    with gr.Row():
        harm_method    = gr.Dropdown(choices=list(HARMONIZE_METHODS.keys()),
                                      value="LAB Color Transfer", label="Method")
        harm_intensity = gr.Slider(0.0,1.0,value=0.7,step=0.05,label="Intensity")
    harm_btn    = gr.Button("Apply Harmonization", variant="secondary")
    harm_out    = gr.Image(label="Harmonized", type="numpy", interactive=False)
    harm_status = gr.Textbox(label="", interactive=False, show_label=False)

    # Step 4: Shadow
    gr.HTML('<div id="section-shadow"></div>')
    gr.Markdown("### 04 — SHADOW SYNTHESIS")
    shadow_method = gr.Dropdown(choices=list(SHADOW_METHODS.keys()),
                                 value="Geometric — instant, CPU-friendly", label="Shadow Method")
    with gr.Row():
        op_sl = gr.Slider(0.1,1.0,value=0.65,step=0.05,label="Opacity")
        ln_sl = gr.Slider(0.3,3.0,value=1.2, step=0.1, label="Length")
        bl_sl = gr.Slider(2,40,value=20,step=1,label="Blur")
    shadow_btn    = gr.Button("Cast Shadow", variant="primary")
    result_out    = gr.Image(label="Final Result", type="numpy", interactive=False)
    shadow_status = gr.Textbox(label="", interactive=False, show_label=False)

    gr.Markdown("**Post-Generation Shadow Edit**")
    darkness_sl   = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                               label="Shadow Darkness (0=none, 1=full)")
    darkness_btn  = gr.Button("Apply Darkness", variant="secondary")
    darkness_out  = gr.Image(label="Adjusted Shadow", type="numpy", interactive=False)
    darkness_status = gr.Textbox(label="", interactive=False, show_label=False)

    gr.Markdown("**Shadow Position Edit**")
    with gr.Row():
        shift_x_sl = gr.Slider(-200, 200, value=0, step=5,
                                label="Shift X (← left / right →)")
        shift_y_sl = gr.Slider(-200, 200, value=0, step=5,
                                label="Shift Y (↑ up / down ↓)")
    shift_dark_sl = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                               label="Shadow Darkness")
    move_btn      = gr.Button("Move Shadow", variant="secondary")
    move_out      = gr.Image(label="Repositioned Shadow", type="numpy", interactive=False)
    move_status   = gr.Textbox(label="", interactive=False, show_label=False)

    # Step 5: A/B Testing
    gr.HTML('<div id="section-ab"></div>')
    gr.Markdown("### 05 — A/B COMPARISON")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Method A**")
            ab_ha = gr.Dropdown(choices=list(HARMONIZE_METHODS.keys()),value="LAB Color Transfer",label="Harmonization A")
            ab_ia = gr.Slider(0.0,1.0,value=0.7,step=0.05,label="Intensity A")
            ab_sa = gr.Dropdown(choices=list(SHADOW_METHODS.keys()),value="Geometric — instant, CPU-friendly",label="Shadow A")
            ab_oa = gr.Slider(0.1,1.0,value=0.65,step=0.05,label="Opacity A")
        with gr.Column():
            gr.Markdown("**Method B**")
            ab_hb = gr.Dropdown(choices=list(HARMONIZE_METHODS.keys()),value="Reinhard Color Transfer",label="Harmonization B")
            ab_ib = gr.Slider(0.0,1.0,value=0.7,step=0.05,label="Intensity B")
            ab_sb = gr.Dropdown(choices=list(SHADOW_METHODS.keys()),value="GPSDiffusion — deep learning, scene-aware",label="Shadow B")
            ab_ob = gr.Slider(0.1,1.0,value=0.65,step=0.05,label="Opacity B")
    ab_btn    = gr.Button("Run A/B Test", variant="primary")
    ab_status = gr.Textbox(label="", interactive=False, show_label=False)
    with gr.Row():
        ab_out_a = gr.Image(label="Result A", type="numpy", interactive=False)
        ab_out_b = gr.Image(label="Result B", type="numpy", interactive=False)

    # ── STEP 6: Evaluation ───────────────────────────────────────
    gr.HTML('<div id="section-evaluation"></div>')
    gr.Markdown("### 06 — EVALUATION")
    gr.Markdown("""
    | Metric | Measures | Direction |
    |---|---|---|
    | NIQE | Overall realism (no-reference) | ↓ lower better |
    | BRISQUE | Blind quality score | ↓ lower better |
    | CLIP Score | Perceptual realism | ↑ higher better |
    | ΔE Color | Color harmonization accuracy | ↓ lower better |
    | Shadow Area % | Shadow coverage | — informational |
    | Shadow Darkness | Shadow intensity | — informational |
    | Shadow Sharpness | Edge quality | — informational |
    """)
    eval_btn    = gr.Button("Run Evaluation", variant="primary")
    eval_output = gr.Textbox(label="Evaluation Results", interactive=False,
                              lines=16, show_label=True)

    # ── EVENTS ───────────────────────────────────────────────────

    extract_mode.change(
        fn=lambda mode: (gr.update(visible="Deep Learning" in mode),
                         gr.update(visible="KNN" in mode)),
        inputs=[extract_mode], outputs=[dl_group, knn_group],
    )

    dl_btn.click(
        fn=do_dl_extract,
        inputs=[fg_upload, dl_method],
        outputs=[dl_out, fg_b64_state, dl_status],
    )

    def on_prefill(fg_np, opacity):
        trimap_np, _, _, msg = auto_prefill_trimap(fg_np)
        if trimap_np is not None:
            h,w = trimap_np.shape
            vis  = np.zeros((h,w,3), dtype=np.uint8)
            vis[trimap_np==255]=[0,200,0]; vis[trimap_np==0]=[200,0,0]; vis[trimap_np==128]=[200,200,0]
            blended = (vis.astype(np.float32)*opacity + fg_np[:,:,:3].astype(np.float32)*(1-opacity)).astype(np.uint8)
        else: blended = None
        return trimap_np, blended, msg

    prefill_btn.click(fn=on_prefill, inputs=[fg_upload, trimap_opacity],
                      outputs=[trimap_state, trimap_editor, prefill_status])

    def update_opacity(fg_np, trimap_np, opacity):
        if trimap_np is None or fg_np is None: return None
        h,w = trimap_np.shape
        vis  = np.zeros((h,w,3), dtype=np.uint8)
        vis[trimap_np==255]=[0,200,0]; vis[trimap_np==0]=[200,0,0]; vis[trimap_np==128]=[200,200,0]
        return (vis.astype(np.float32)*opacity + fg_np[:,:,:3].astype(np.float32)*(1-opacity)).astype(np.uint8)

    trimap_opacity.release(fn=update_opacity,
                           inputs=[fg_upload, trimap_state, trimap_opacity],
                           outputs=[trimap_editor])

    def image_to_trimap(editor_data, fallback):
        try:
            if editor_data is not None:
                img = editor_data.get("composite", editor_data.get("background")) if isinstance(editor_data,dict) else editor_data
                if img is not None:
                    img = np.array(img)[:,:,:3]
                    trimap = np.full(img.shape[:2], 128, dtype=np.uint8)
                    trimap[(img[:,:,1]>150)&(img[:,:,0]<100)] = 255
                    trimap[(img[:,:,0]>150)&(img[:,:,1]<100)] = 0
                    trimap[(img[:,:,0]>150)&(img[:,:,1]>150)] = 128
                    return trimap
        except Exception as e: print(f"[Trimap] {e}")
        return fallback

    def on_knn(fg_np, editor_data, trimap_np_state):
        trimap = image_to_trimap(editor_data, trimap_np_state)
        if fg_np is None: return None,None,"Upload foreground first."
        if trimap is None: return None,None,"Run Auto-Prefill first."
        return run_knn_matting(fg_np, trimap)

    knn_btn.click(fn=on_knn, inputs=[fg_upload, trimap_editor, trimap_state],
                  outputs=[knn_out, fg_b64_state, knn_status])

    def on_composite(bg_np, fg_b64, x, y, scale):
        return do_composite(bg_np, fg_b64, x, y, scale)

    comp_btn.click(fn=on_composite, inputs=[bg_input, fg_b64_state, x_sl, y_sl, s_sl],
                   outputs=[comp_preview, pos_state, comp_status])

    for sl in [x_sl, y_sl, s_sl]:
        sl.release(fn=on_composite, inputs=[bg_input, fg_b64_state, x_sl, y_sl, s_sl],
                   outputs=[comp_preview, pos_state, comp_status])

    harm_btn.click(
        fn=do_harmonize,
        inputs=[comp_preview, fg_b64_state, pos_state, harm_method, harm_intensity],
        outputs=[harm_out, harmonized_state, harm_status],
    )


    shadow_btn.click(
        fn=do_shadow,
        inputs=[harmonized_state, comp_preview, fg_b64_state,
                pos_state, bg_input, shadow_method, op_sl, ln_sl, bl_sl],
        outputs=[result_out, shadow_status],
    )

    ab_btn.click(
        fn=do_ab,
        inputs=[comp_preview, fg_b64_state, pos_state, bg_input,
                ab_ha, ab_ia, ab_sa, ab_oa, ab_hb, ab_ib, ab_sb, ab_ob],
        outputs=[ab_out_a, ab_out_b, ab_status],
    )

    darkness_btn.click(
        fn=adjust_shadow_darkness,
        inputs=[result_out, comp_preview, darkness_sl, pos_state],
        outputs=[darkness_out, darkness_status],
    )

    # Also live update on slider release
    darkness_sl.release(
        fn=adjust_shadow_darkness,
        inputs=[result_out, comp_preview, darkness_sl, pos_state],
        outputs=[darkness_out, darkness_status],
    )

    move_btn.click(
        fn=move_shadow,
        inputs=[result_out, comp_preview, shift_x_sl, shift_y_sl, shift_dark_sl],
        outputs=[move_out, move_status],
    )
    # Live update on slider release
    for sl in [shift_x_sl, shift_y_sl, shift_dark_sl]:
        sl.release(
            fn=move_shadow,
            inputs=[result_out, comp_preview, shift_x_sl, shift_y_sl, shift_dark_sl],
            outputs=[move_out, move_status],
        )

    eval_btn.click(
        fn=lambda a, b, comp, pos: run_evaluation(a, b, comp, pos),
        inputs=[ab_out_a, ab_out_b, comp_preview, pos_state],
        outputs=[eval_output],
    )



if __name__ == "__main__":
    demo.launch(share=True)
