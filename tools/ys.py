好的，這是 ys.py 的完整內容（可直接複製貼上使用）：

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ys.py  —  YOLOv7 Short Sweep
目的：不改 train.py，用最少參數做常用 sweep（workers × batch 等），自動挑最佳組合。
特色（簡單版但夠用）：
  1) 自動偵測 GPU（H100/5090/4090...）與 /dev/shm，實驗名短而易讀 (如 KATN-G4090-bs512-w12-RQ)。
  2) 兩種 profile：--fast（預設）/ --regular，會依 GPU 選合適網格（可在 sweep_profiles.yaml 改）。
  3) 一次 sweep 多組：workers、batch_size、rect/quad、增強關閉（A0），自動解析 s/it、GPU 利用率/功耗/VRAM。
  4) OOM 自動回退（每次 -64）直到 --min-batch。
  5) 依 batch 線性放大 hyp 的 lr0（可用 --no-scale-lr 關閉）。
  6) 產出 CSV，並印出最佳建議指令（可直接複製）。

最小用法：
  python ys.py --svr runpod \
    --train train.py --data data/coco.yaml --cfg cfg/training/yolov7-tiny.yaml \
    --weights yolov7-tiny.pt --hyp data/hyp.scratch.tiny.bs384.yaml --img 320 --epochs 1

可選：
  --regular（取代 --fast）
  --bs 384,512,640   （覆寫 profile 的 batch_sizes）
  --w  8,12,16       （覆寫 profile 的 workers）
"""

import argparse, os, sys, time, subprocess, threading, shutil, re, csv, statistics, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import yaml
except Exception:
    yaml = None

SIT_RE = re.compile(r'([0-9]*\.?[0-9]+)\s*s/it')
NB_RE  = re.compile(r'(\d+)\s*/\s*(\d+)')
IMAGES_RE = re.compile(r'(\d+)\s+images.*(train|training)', re.IGNORECASE)
OOM_RE = re.compile(r'out of memory', re.IGNORECASE)

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

def detect_gpu() -> Dict[str, Any]:
    info = {"name":"UnknownGPU","vram_mib":None,"short":"UNK"}
    if shutil.which("nvidia-smi"):
        out = read_cmd(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"])
        line = out.splitlines()[0].strip() if out else ""
        if line:
            try:
                name, vram = [x.strip() for x in line.split(",")]
                info["name"]=name; info["vram_mib"]=int(vram)
            except: pass
    n = info["name"].upper()
    if   "H100" in n: info["short"]="H100"
    elif "B200" in n: info["short"]="B200"
    elif "5090" in n: info["short"]="G5090"
    elif "4090" in n: info["short"]="G4090"
    elif "A100" in n: info["short"]="A100"
    else:
        gb = int(info["vram_mib"]/1024) if info["vram_mib"] else 0
        info["short"]=(n.split()[0][:1] if n else "G")+str(gb)
    return info

def check_shm() -> int:
    try:
        s=os.statvfs("/dev/shm"); return int(s.f_bsize*s.f_bavail/(1024*1024))
    except Exception: return -1

def cool_prefix(seed: str) -> str:
    pool=["KATN","BLTZ","VIPR","NOVA","ZEUS","RAGN","FURY","RAPX","BLAZ","PHNX","DRGN","TGRS"]
    h=int(hashlib.md5(seed.encode()).hexdigest(),16)
    return pool[h%len(pool)]

class GPUMonitor(threading.Thread):
    def __init__(self, interval_s=1.0, idx=0):
        super().__init__(daemon=True); self.interval_s=interval_s; self.idx=idx
        self.util=[]; self.pwr=[]; self.mem=[]; self._stop=threading.Event()
        self.proc=None; self.ok=shutil.which("nvidia-smi") is not None
    def run(self):
        if not self.ok: return
        cmd=["nvidia-smi",f"--id={self.idx}","--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits",f"--loop-ms={int(self.interval_s*1000)}"]
        self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in self.proc.stdout:
            if self._stop.is_set(): break
            try:
                u,p,m=[x.strip() for x in line.strip().split(",")]
                if u.isdigit(): self.util.append(int(u))
                try: self.pwr.append(float(p))
                except: pass
                if m.isdigit(): self.mem.append(int(m))
            except: pass
    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try: self.proc.terminate()
            except: pass
    def summary(self):
        def avg(v): return round(sum(v)/len(v),2) if v else None
        return {"gpu_util_avg":avg(self.util),"power_avg_w":avg(self.pwr),"vram_avg_mib":avg(self.mem),"samples":len(self.util)}

def flags_str(rect,quad,aug_off):
    lbl=""
    if rect: lbl+="R"
    if quad: lbl+="Q"
    if aug_off: lbl+="A0"
    return lbl or "DEF"

def scale_lr0(hyp_path: Optional[str], batch: int, base_bs: int = 64) -> Optional[str]:
    if not hyp_path or not os.path.isfile(hyp_path) or yaml is None:
        return hyp_path
    try:
        with open(hyp_path,"r",encoding="utf-8") as f: hyp=yaml.safe_load(f)
        lr0=float(hyp.get("lr0",0.01))* (batch/base_bs)
        hyp2=dict(hyp); hyp2["lr0"]=float(lr0)
        out=Path(hyp_path).with_suffix("") ; out=Path(str(out)+f".lr{lr0:.5f}.yaml")
        with open(out,"w",encoding="utf-8") as g: yaml.safe_dump(hyp2,g,sort_keys=False,allow_unicode=True)
        return str(out)
    except: return hyp_path

def aug_override(hyp_path: Optional[str], off: bool) -> Optional[str]:
    if not hyp_path or not os.path.isfile(hyp_path) or yaml is None:
        return hyp_path
    try:
        with open(hyp_path,"r",encoding="utf-8") as f: hyp=yaml.safe_load(f)
        hyp2=dict(hyp)
        if off:
            for k in ["mosaic","mixup","degrees","shear","perspective","translate","scale"]:
                if k in hyp2: hyp2[k]=0.0
        out=Path(hyp_path).with_name(Path(hyp_path).stem+(".aug0.yaml" if off else ".aug.yaml"))
        with open(out,"w",encoding="utf-8") as g: yaml.safe_dump(hyp2,g,sort_keys=False,allow_unicode=True)
        return str(out)
    except: return hyp_path

def build_cmd(opt, workers, batch, rect, quad, aug_off, hyp_file, name):
    cmd=[sys.executable,opt.train,"--img-size",str(opt.img),"--batch-size",str(batch),"--epochs",str(opt.epochs),
         "--data",opt.data,"--cfg",opt.cfg,"--weights",opt.weights,"--device",opt.device,
         "--workers",str(workers),"--project",opt.project,"--name",name,"--exist-ok"]
    if opt.cache: cmd+=["--cache-images"]
    if rect: cmd+=["--rect"]
    if quad: cmd+=["--quad"]
    if opt.save_period is not None: cmd+=["--save_period",str(opt.save_period)]
    if hyp_file: cmd+=["--hyp",hyp_file]
    return cmd

def run_one(opt, gpu, logdir: Path, workers, batch, rect, quad, aug_off) -> Dict[str,Any]:
    hyp1=aug_override(opt.hyp, aug_off) if opt.hyp else None
    hyp2=scale_lr0(hyp1 or opt.hyp, batch) if opt.scale_lr else (hyp1 or opt.hyp)

    prefix=cool_prefix(f"{gpu['name']}-{workers}-{batch}-{rect}-{quad}-{aug_off}")
    flags=flags_str(rect,quad,aug_off)
    ts=datetime.now().strftime("%m%d%H%M")
    name=f"{prefix}-{gpu['short']}-bs{batch}-w{workers}-{flags}-{ts}"
    logfile=logdir/f"{name}.log"
    cmd=build_cmd(opt,workers,batch,rect,quad,aug_off,hyp2,name)

    env=os.environ.copy()
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        env.setdefault(k,"1")

    mon=GPUMonitor(1.0,idx=int(opt.gpu_index)); mon.start()

    sit_vals=[]; nb=None; t0e=None; t1e=None; saw_oom=False
    t0=time.time()
    with open(logfile,"w",encoding="utf-8") as f:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        try:
            for raw in p.stdout:
                line=raw.rstrip("\n"); f.write(line+"\n")
                if "s/it" in line or "Epoch" in line: print(line)
                if OOM_RE.search(line): saw_oom=True
                m=SIT_RE.search(line)
                if m:
                    try:
                        v=float(m.group(1))
                        if v>0: sit_vals.append(v)
                    except: pass
                m2=NB_RE.search(line)
                if m2:
                    try:
                        cur,den=int(m2.group(1)),int(m2.group(2)); nb=den
                        if cur==0 and t0e is None: t0e=time.time()
                        if nb is not None and cur==nb and t0e and t1e is None: t1e=time.time()
                    except: pass
        finally:
            p.wait(); rc=p.returncode; mon.stop()
    t1=time.time(); wall=t1-t0; gsum=mon.summary()

    if sit_vals: s_it=round(statistics.median(sit_vals),3); method="median(s/it)"
    elif t0e and t1e and nb: s_it=round((t1e-t0e)/nb,3); method="epoch0_wall/nb"
    elif nb: s_it=round(wall/nb,3); method="wall/nb"
    else: s_it=None; method="unknown"

    return {
        "exp":name,"server":opt.svr,"gpu":gpu["name"],"gshort":gpu["short"],"vram_mib":gpu["vram_mib"],
        "w":workers,"bs":batch,"img":opt.img,"rect":int(rect),"quad":int(quad),"aug_off":int(aug_off),
        "epochs":opt.epochs,"s_it":s_it,"method":method,"wall_s":round(wall,3),"nb":nb,
        "gpu_util":gsum.get("gpu_util_avg"),"pwr_w":gsum.get("power_avg_w"),"vram_mib_avg":gsum.get("vram_avg_mib"),
        "log":str(logfile),"rc":rc,"oom":int(saw_oom),"hyp":hyp2 or opt.hyp
    }

def load_profiles(path: str) -> dict:
    if yaml is None: return {}
    if not os.path.isfile(path): return {}
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f) or {}

def main():
    ap=argparse.ArgumentParser(description="YOLOv7 Short Sweep (ys.py)")
    ap.add_argument("--svr",type=str,default="unspecified",help="server 標記：runpod / vast.ai / local")
    ap.add_argument("--fast",action="store_true",default=True,help="快測（預設）")
    ap.add_argument("--regular",action="store_true",help="常規訓練 sweep")
    ap.add_argument("--train",type=str,default="train.py")
    ap.add_argument("--data",type=str,default="data/coco.yaml")
    ap.add_argument("--cfg",type=str,default="cfg/training/yolov7-tiny.yaml")
    ap.add_argument("--weights",type=str,default="yolov7-tiny.pt")
    ap.add_argument("--hyp",type=str,default="data/hyp.scratch.tiny.bs384.yaml")
    ap.add_argument("--img",type=int,default=320)
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--device",type=str,default="0")
    ap.add_argument("--project",type=str,default="runs/feasibility")
    ap.add_argument("--save_period",type=int,default=5)
    ap.add_argument("--cache",action="store_true",default=True)
    ap.add_argument("--gpu_index",type=int,default=0)
    ap.add_argument("--scale-lr",dest="scale_lr",action="store_true",default=True)
    ap.add_argument("--no-scale-lr",dest="scale_lr",action="store_false")
    ap.add_argument("--profiles",type=str,default="sweep_profiles.yaml")
    ap.add_argument("--bs",type=str,default=None,help="覆寫 batch_sizes，例：384,512,640")
    ap.add_argument("--w",type=str,default=None,help="覆寫 workers，例：8,12,16")
    ap.add_argument("--csv",type=str,default=None)
    ap.add_argument("--min-batch",type=int,default=128)
    args=ap.parse_args()

    # fast/regular 二選一：若指定 --regular，則關閉 fast
    if args.regular: args.fast=False

    gpu=detect_gpu(); shm=check_shm()
    print(f"[{now()}] Server={args.svr} | GPU={gpu['name']} ({gpu['short']}) | /dev/shm={shm} MiB")
    if shm!=-1 and shm<8192:
        print("⚠️  /dev/shm < 8 GiB，建議 docker 加：--shm-size=16g --ipc=host")

    date_tag=datetime.now().strftime("%Y%m%d-%H%M")
    base_out=Path(args.project)/f"sweeps-{date_tag}"
    logdir=base_out/"logs"; logdir.mkdir(parents=True,exist_ok=True)
    csv_path=Path(args.csv) if args.csv else (base_out/"results.csv")

    # 讀 profile
    prof=load_profiles(args.profiles)
    key=None
    # 自動選對應 GPU 的 profile key
    if args.fast:
        for cand in (f"{gpu['short']}-fast","generic-fast"):
            if cand in prof: key=cand; break
    else:
        for cand in (f"{gpu['short']}-regular","generic-regular"):
            if cand in prof: key=cand; break
    if not key:
        print("❌ 找不到合適的 profile，請檢查 sweep_profiles.yaml"); sys.exit(2)
    p=prof[key]

    # 允許覆寫 workers/batch_sizes
    workers = [int(x) for x in (args.w.split(",") if args.w else p.get("workers",[8,12]))]
    batches = [int(x) for x in (args.bs.split(",") if args.bs else p.get("batch_sizes",[384,512]))]
    rect    = bool((p.get("rect",[1 if args.fast else 0])[0]) if isinstance(p.get("rect"),list) else p.get("rect",1))
    quad    = bool((p.get("quad",[1])[0]) if isinstance(p.get("quad"),list) else p.get("quad",1))
    aug_opts= p.get("aug_off",[1,0] if args.fast else [0])
    if isinstance(aug_opts,int): aug_opts=[aug_opts]

    # 產生實驗組（簡化：不掃 multi-scale/image-weights）
    grid=[]
    for bs in sorted(batches, reverse=True):
        for wv in workers:
            for ao in aug_opts:
                grid.append(dict(bs=bs, w=wv, rect=rect, quad=quad, aug_off=bool(int(ao))))

    results=[]
    for g in grid:
        # OOM 回退：bs, bs-64, bs-128, ...
        seq=[g["bs"]]; b=g["bs"]-64
        while b>=args.min_batch:
            seq.append(b); b-=64
        ok=False
        for bs in seq:
            r=run_one(args,gpu,logdir,g["w"],bs,g["rect"],g["quad"],g["aug_off"])
            results.append(r)
            if r["rc"]==0 and not r["oom"]:
                ok=True; break
            else:
                print(f"⚠️  rc={r['rc']} OOM={r['oom']} → 試較小 batch...")
        if not ok: print("❌ 這組全失敗，略過")

    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results: writer.writerow(r)

    # 挑最佳（最小 s/it）
    okr=[r for r in results if r["s_it"] is not None and r["rc"]==0]
    best=min(okr,key=lambda x:x["s_it"]) if okr else None

    print("\n========== Summary ==========\n")
    hdr=["exp","gshort","w","bs","rect","quad","aug_off","s_it","gpu_util","pwr_w","vram_mib_avg","wall_s","log"]
    print(" | ".join(f"{h:>10}" for h in hdr))
    for r in results:
        print(" | ".join(f"{str(r.get(h))[:10]:>10}" for h in hdr))
    print(f"\nCSV: {csv_path}")
    if best:
        flags=[]; 
        if best["rect"]: flags+=["--rect"]
        if best["quad"]: flags+=["--quad"]
        cmd=f"""{sys.executable} {args.train} --img-size {args.img} --batch-size {best['bs']} --epochs {args.epochs} \
--data {args.data} --cfg {args.cfg} --weights {args.weights} --device {args.device} --workers {best['w']} \
--project {args.project} --name {best['exp']}-FINAL --exist-ok {'--cache-images' if args.cache else ''} {' '.join(flags)}"""
        print("\n🏆 Best by s/it：", best["exp"], "s/it=", best["s_it"], " GPU%~", best["gpu_util"])
        print("\n🔧 建議最佳指令：\n"+cmd+"\n")

if __name__=="__main__":
    main()