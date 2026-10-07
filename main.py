# main.py
import math, re, random, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

SPECIAL = ["<pad>","<bos>","<eos>","<el>","<click>","<type>","<select>","<stop>"]
INTERACTIVE = {"a","button","input","select","textarea"}
SKIP = {"script","style","head","meta","link","svg","path"}

@dataclass
class Axiom1Config:
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    n_memory_tokens: int = 16
    window_size: int = 32
    max_seq_len: int = 192
    max_action_len: int = 16
    dropout: float = 0.1
    ffn_mult: int = 4
    rope_theta: float = 10000.0
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    el_id: int = 4

def build_tokenizer(texts, vocab_size=2000):
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    t.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<unk>"]+SPECIAL, show_progress=False))
    return t

def tid(tok, s): return tok.token_to_id(s)
def enc(tok, s): return tok.encode(s).ids

# ---------- data ----------
def fetch_data(n=800):
    from datasets import load_dataset
    return load_dataset("osunlp/Mind2Web", split=f"train[:{n}]")

def _walk(node, parts, els):
    from bs4 import Tag
    if not isinstance(node, Tag): return
    if node.name in SKIP: return
    if node.name in INTERACTIVE:
        t = re.sub(r"\s+"," ",node.get_text(" ",strip=True)).strip()[:80]
        els.append({"tag":node.name,"text":t})
        parts.append("<el>")
        if t: parts.append(t)
        parts.append("</el>"); return
    for c in node.children: _walk(c, parts, els)

def parse_one(ex, tok, cfg):
    from bs4 import BeautifulSoup
    html = ex.get("cleaned_html") or ""
    q = ex.get("confirmed_task") or ""
    if not html or not q: return None
    soup = BeautifulSoup(html,"lxml")
    parts, els = [], []
    _walk(soup, parts, els)
    if not els: return None
    text = re.sub(r"\s+"," "," ".join(parts))[:3000]
    pids = enc(tok,text)[:cfg.max_seq_len-32]
    eltok = tid(tok,"<el>")
    elpos = [i for i,t in enumerate(pids) if t==eltok]
    if not elpos: return None
    qids = enc(tok,q)
    ids = pids+qids
    pm = [1]*len(pids)+[0]*len(qids)
    am = [1]*len(ids)
    bos,el,cl,ty,st = tid(tok,"<bos>"),tid(tok,"<el>"),tid(tok,"<click>"),tid(tok,"<type>"),tid(tok,"<stop>")
    act,tgt = [bos],[-1]
    for a in (ex.get("actions") or [])[:4]:
        if not isinstance(a,dict): continue
        op = a.get("operation") or {}
        on = str(op.get("op","") if isinstance(op,dict) else op).upper()
        val = str(op.get("value","") if isinstance(op,dict) else "")
        e = None
        ts = a.get("target") or ""
        if isinstance(ts,str) and ts.strip():
            try:
                n = BeautifulSoup(ts,"lxml").find()
                if n:
                    txt = re.sub(r"\s+"," ",n.get_text(" ",strip=True)).strip()[:80]
                    for i,ee in enumerate(els):
                        if ee["tag"]==n.name and txt and txt in ee["text"]:
                            e = i+1; break
            except Exception: pass
        pos = elpos[e-1] if e and 1<=e<=len(elpos) else -1
        if "CLICK" in on: act+=[cl,el]; tgt+=[-1,pos]
        elif "TYPE" in on:
            act+=[ty,el]; tgt+=[-1,pos]
            v = enc(tok," "+val[:30]); act+=v; tgt+=[-1]*len(v)
        elif "STOP" in on: break
    act.append(st); tgt.append(-1)
    if len(act)>cfg.max_action_len:
        act = act[:cfg.max_action_len-1]+[st]
        tgt = tgt[:cfg.max_action_len-1]+[-1]
    return {"input_ids":ids,"attention_mask":am,"page_mask":pm,"action_ids":act,"elem_targets":tgt}

def build_examples(ds, tok, cfg):
    out = []
    for ex in ds:
        try:
            p = parse_one(ex, tok, cfg)
            if p: out.append(p)
        except Exception: pass
    return out

# ---------- dataset ----------
class TrajectoryDataset(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        return {k: torch.tensor(v, dtype=torch.long) for k,v in self.ex[i].items()}

def collate(b, pad_id=0):
    mi = max(x["input_ids"].numel() for x in b)
    ma = max(x["action_ids"].numel() for x in b)
    out = {k: [] for k in ["input_ids","attention_mask","page_mask","action_ids","elem_targets"]}
    for x in b:
        pi = mi - x["input_ids"].numel(); pa = ma - x["action_ids"].numel()
        out["input_ids"].append(F.pad(x["input_ids"], (0,pi), value=pad_id))
        out["attention_mask"].append(F.pad(x["attention_mask"], (0,pi)))
        out["page_mask"].append(F.pad(x["page_mask"], (0,pi)))
        out["action_ids"].append(F.pad(x["action_ids"], (0,pa), value=pad_id))
        out["elem_targets"].append(F.pad(x["elem_targets"], (0,pa), value=-1))
    return {k: torch.stack(v) for k,v in out.items()}

# ---------- model ----------
def _rope(dim, n, theta, dev):
    f = 1.0/(theta**(torch.arange(0,dim,2,device=dev).float()/dim))
    a = torch.outer(torch.arange(n,device=dev).float(), f)
    return a.cos().repeat_interleave(2,-1), a.sin().repeat_interleave(2,-1)

def _rh(x):
    a,b = x.chunk(2,-1); return torch.cat([-b,a],-1)

def _ra(x,c,s): return x*c[None,None]+_rh(x)*s[None,None]

def _ea(q,k,v,gm,w,kpm,dp):
    B,H,S,D = q.shape
    i = torch.arange(S,device=q.device)
    loc = (i[:,None]-i[None,:]).abs() <= w
    allow = gm[:,None,:] | gm[:,:,None] | loc[None]
    if kpm is not None: allow = allow & (~kpm)[:,None,:]
    bias = torch.zeros_like(allow,dtype=q.dtype).masked_fill(~allow,float("-inf"))
    return F.scaled_dot_product_attention(q,k,v,attn_mask=bias.unsqueeze(1),dropout_p=dp)

class EncBlock(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c
        s.n1=nn.LayerNorm(c.d_model); s.n2=nn.LayerNorm(c.d_model)
        s.qkv=nn.Linear(c.d_model,3*c.d_model); s.pr=nn.Linear(c.d_model,c.d_model)
        s.ff=nn.Sequential(nn.Linear(c.d_model,c.ffn_mult*c.d_model),nn.GELU(),
                          nn.Linear(c.ffn_mult*c.d_model,c.d_model))
        s.d=nn.Dropout(c.dropout)
    def forward(s,x,co,si,gm,kpm):
        h=s.n1(x); B,S,D=h.shape; H=s.c.n_heads
        qkv=s.qkv(h).view(B,S,3,H,D//H).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=_ra(q,co,si); k=_ra(k,co,si)
        dp=s.c.dropout if s.training else 0.0
        a=_ea(q,k,v,gm,s.c.window_size,kpm,dp).transpose(1,2).reshape(B,S,D)
        x=x+s.d(s.pr(a)); x=x+s.d(s.ff(s.n2(x))); return x

class Encoder(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c
        s.tok=nn.Embedding(c.vocab_size,c.d_model)
        s.mem=nn.Parameter(torch.randn(1,c.n_memory_tokens,c.d_model)*0.02)
        s.bl=nn.ModuleList([EncBlock(c) for _ in range(c.n_encoder_layers)])
        s.n=nn.LayerNorm(c.d_model); s.d=nn.Dropout(c.dropout)
        co,si=_rope(c.d_model//c.n_heads,c.max_seq_len+c.n_memory_tokens,c.rope_theta,torch.device("cpu"))
        s.register_buffer("co",co,persistent=False)
        s.register_buffer("si",si,persistent=False)
    def forward(s,ids,am,plen):
        B,L=ids.shape; M=s.c.n_memory_tokens
        x=s.d(s.tok(ids))
        x=torch.cat([s.mem.expand(B,-1,-1),x],1); S=x.size(1)
        co=s.co[:S].to(x.dtype); si=s.si[:S].to(x.dtype)
        ones=torch.ones(B,M,device=x.device,dtype=am.dtype)
        kpm=(torch.cat([ones,am],1)==0)
        gm=torch.zeros(B,S,dtype=torch.bool,device=x.device); gm[:,:M]=True
        if plen<L: gm[:,M+plen:]=True
        for b in s.bl: x=b(x,co,si,gm,kpm)
        x=s.n(x); return x[:,:M],x[:,M:M+plen]

class DecBlock(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c
        s.n1=nn.LayerNorm(c.d_model); s.n2=nn.LayerNorm(c.d_model); s.n3=nn.LayerNorm(c.d_model)
        s.qkv=nn.Linear(c.d_model,3*c.d_model); s.pr=nn.Linear(c.d_model,c.d_model)
        s.cq=nn.Linear(c.d_model,c.d_model); s.ckv=nn.Linear(c.d_model,2*c.d_model)
        s.cp=nn.Linear(c.d_model,c.d_model)
        s.ff=nn.Sequential(nn.Linear(c.d_model,c.ffn_mult*c.d_model),nn.GELU(),
                          nn.Linear(c.ffn_mult*c.d_model,c.d_model))
        s.d=nn.Dropout(c.dropout)
    def forward(s,x,m,co,si):
        B,S,D=x.shape; H=s.c.n_heads; dp=s.c.dropout if s.training else 0.0
        h=s.n1(x)
        qkv=s.qkv(h).view(B,S,3,H,D//H).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=_ra(q,co,si); k=_ra(k,co,si)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=dp).transpose(1,2).reshape(B,S,D)
        x=x+s.d(s.pr(a))
        h=s.n2(x)
        q=s.cq(h).view(B,S,H,D//H).transpose(1,2)
        _,Sm,_=m.shape
        kv=s.ckv(m).view(B,Sm,2,H,D//H).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,kv[0],kv[1],dropout_p=dp).transpose(1,2).reshape(B,S,D)
        x=x+s.d(s.cp(a)); x=x+s.d(s.ff(s.n3(x))); return x

class Decoder(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c
        s.tok=nn.Embedding(c.vocab_size,c.d_model)
        s.bl=nn.ModuleList([DecBlock(c) for _ in range(c.n_decoder_layers)])
        s.n=nn.LayerNorm(c.d_model)
        s.head=nn.Linear(c.d_model,c.vocab_size,bias=False); s.head.weight=s.tok.weight
        s.pq=nn.Linear(c.d_model,c.d_model,bias=False); s.pk=nn.Linear(c.d_model,c.d_model,bias=False)
        s.d=nn.Dropout(c.dropout)
        co,si=_rope(c.d_model//c.n_heads,c.max_action_len,c.rope_theta,torch.device("cpu"))
        s.register_buffer("co",co,persistent=False)
        s.register_buffer("si",si,persistent=False)
    def forward(s,a,m,ps,pm):
        B,A=a.shape
        x=s.d(s.tok(a))
        co=s.co[:A].to(x.dtype); si=s.si[:A].to(x.dtype)
        for b in s.bl: x=b(x,m,co,si)
        h=s.n(x)
        lg=s.head(h)
        q=s.pq(h); k=s.pk(ps)
        sc=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.size(-1))
        if pm is not None: sc=sc.masked_fill(~pm[:,None,:],float("-inf"))
        return lg,sc

class Axiom1(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c
        s.enc=Encoder(c); s.dec=Decoder(c)
    def forward(s,ids,am,pm,plen,act):
        m,ps=s.enc(ids,am,plen)
        return s.dec(act[:,:-1],m,ps,pm[:,:plen])
    @torch.no_grad()
    def generate(s,ids,am,pm,plen,max_len=None):
        s.eval()
        max_len=max_len or s.c.max_action_len
        m,ps=s.enc(ids,am,plen); pmt=pm[:,:plen]; B=ids.size(0)
        toks=torch.full((B,1),s.c.bos_id,dtype=torch.long,device=ids.device)
        positions=[]
        for _ in range(max_len-1):
            lg,pt=s.dec(toks,m,ps,pmt)
            nxt=lg[:,-1,:].argmax(-1,keepdim=True)
            p=pt[:,-1,:].argmax(-1)
            positions.append(p if (nxt==s.c.el_id).any() else torch.full_like(p,-1))
            toks=torch.cat([toks,nxt],1)
            if (nxt==s.c.eos_id).all(): break
        pos=torch.stack(positions,1) if positions else torch.full((B,0),-1,dtype=torch.long,device=ids.device)
        return toks,pos

# ---------- train / serve ----------
def train(cfg, examples, epochs=3, bs=8, lr=1e-3, warmup=20, device="cuda", ckpt="axiom1.pt"):
    model=Axiom1(cfg).to(device)
    loader=DataLoader(TrajectoryDataset(examples), batch_size=bs, shuffle=True,
                      collate_fn=lambda b: collate(b, cfg.pad_id), drop_last=False)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    total=max(1,len(loader)*epochs)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=total,
        pct_start=max(1e-3,warmup/total))
    use_cuda=device=="cuda"
    amp=torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler=torch.amp.GradScaler(device,enabled=(use_cuda and amp==torch.float16))
    model.train(); step=0
    for ep in range(epochs):
        for b in loader:
            ids=b["input_ids"].to(device); am=b["attention_mask"].to(device)
            pm=b["page_mask"].to(device); act=b["action_ids"].to(device)
            et=b["elem_targets"].to(device)
            plen=int(pm.sum(1).max().item())
            with torch.amp.autocast(device,dtype=amp,enabled=use_cuda):
                lg,pt=model(ids,am,pm,plen,act)
                lt=F.cross_entropy(lg.reshape(-1,lg.size(-1)),act[:,1:].reshape(-1),
                                   ignore_index=cfg.pad_id,label_smoothing=0.05)
                lp=F.cross_entropy(pt.reshape(-1,pt.size(-1)),et[:,1:].reshape(-1),ignore_index=-1)
                loss=lt+lp
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step%20==0:
                print(f"ep{ep} step{step}/{total} loss{loss.item():.3f} tok{lt.item():.3f} ptr{lp.item():.3f}")
            step+=1
    torch.save({"model":model.state_dict(),"cfg":asdict(cfg)},ckpt)
    print(f"saved {ckpt}")
    return model

def serve(model, tok, examples, device="cuda", n=5):
    model.eval()
    for i in range(min(n,len(examples))):
        ex=examples[i]
        ids=torch.tensor([ex["input_ids"]],device=device)
        am=torch.tensor([ex["attention_mask"]],device=device)
        pm=torch.tensor([ex["page_mask"]],device=device)
        plen=int(pm.sum().item())
        toks,pos=model.generate(ids,am,pm,plen)
        print(f"sample {i}")
        print("  gen  :",toks[0].tolist())
        print("  text :",tok.decode(toks[0].tolist(),skip_special_tokens=False))
        print("  elem :",pos[0].tolist())
        print("  gold :",ex["action_ids"])
        print("  goldE:",ex["elem_targets"])

# ---------- entry ----------
if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    ds = fetch_data(800)
    print(f"loaded {len(ds)}")
    corpus = [(e.get("cleaned_html") or "")[:2000] for e in ds]
    corpus += [e.get("confirmed_task") or "" for e in ds]
    tok = build_tokenizer(corpus, vocab_size=2000)
    V = tok.get_vocab_size()
    print(f"vocab {V}")
    cfg = Axiom1Config(vocab_size=V, pad_id=tid(tok,"<pad>"), bos_id=tid(tok,"<bos>"),
                       eos_id=tid(tok,"<eos>"), el_id=tid(tok,"<el>"))
    examples = build_examples(ds, tok, cfg)
    print(f"parsed {len(examples)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(cfg, examples, epochs=3, bs=8, device=device, ckpt="axiom1.pt")
    serve(model, tok, examples, device=device, n=5)
