"""Rebuild submission plot from the bundled submission metrics (no network access)."""
from pathlib import Path
import re,json,csv,hashlib
P=Path(__file__).resolve().parents[1];S=P/'data'
ordered=list(csv.DictReader((S/'submission_curve.csv').open(newline='',encoding='utf-8')))
for r in ordered:
 r['file_number']=int(r['file_number'])
 n=r['file_number']
 r['plot_x']=n if n==1 else n-2 if n<32 else n-3
 for key in ['RankIC','IR']:
  r[key]=float(r[key])
ordered.sort(key=lambda r:r['plot_x'])
unique={r['file_number']:r for r in ordered}
assert len(unique)==len(ordered)
meta={'input':'data/submission_curve.csv','plotted_points':len(ordered)}

# Native TikZ coordinates in cm, keeping the full paper self-contained.
lines=[r'\begin{tikzpicture}[font=\footnotesize,line join=round]']
def X(x): return (x-1)*.42
for key,base,low,high,ticks,title,color in [
 ('RankIC',6.7,.057,.094,[.06,.07,.08,.09],'(a) 公榜 RankIC','rhBase'),
 ('IR',.7,-1.15,7.05,[0,2,4,6],'(b) 公榜组合 IR','rhFusion')]:
 def Y(y):return base+(y-low)/(high-low)*4.5
 lines.append(r'\node[anchor=west,font=\small\bfseries] at (0,'+str(base+4.9)+'){'+title+'};')
 for v in ticks:
  y=Y(v);lab=f'{v:.4f}' if key=='RankIC' else str(v)
  lines.append(rf'\draw[black!12] (0,{y:.4f}) -- (13.44,{y:.4f});')
  lines.append(rf'\node[anchor=east] at (-.15,{y:.4f}) {{{lab}}};')
 lines.append(rf'\draw[black!50] (0,{base}) -- (13.44,{base});')
 for r in ordered:
  x=X(r['plot_x']);y=Y(r[key])
  lines.append(rf'\fill[{color}] ({x:.4f},{y:.4f}) circle (1.5pt);')
 # Shape-preserving cubic Hermite interpolation (equal x spacing).
 # Zero slopes at extrema avoid invented peaks between actual submissions.
 pts=[(X(r['plot_x']),Y(r[key])) for r in ordered]
 h=.42;ds=[(pts[i+1][1]-pts[i][1])/h for i in range(len(pts)-1)]
 slopes=[ds[0]]+[0 if a*b<=0 else 2*a*b/(a+b) for a,b in zip(ds,ds[1:])]+[ds[-1]]
 path=f"({pts[0][0]:.4f},{pts[0][1]:.4f})"
 for i,((x0,y0),(x1,y1)) in enumerate(zip(pts,pts[1:])):
  path+=f" .. controls ({x0+h/3:.4f},{y0+slopes[i]*h/3:.4f}) and ({x1-h/3:.4f},{y1-slopes[i+1]*h/3:.4f}) .. ({x1:.4f},{y1:.4f})"
 lines.append(rf'\draw[{color},line width=.8pt] '+path+';')
 if key=='IR':
  for v in [1]+list(range(2,33,2)):
   x=X(v);lines.append(rf'\draw[black!50] ({x},{base}) -- ++(0,-.08) node[below] {{{v}}};')
  lines.append(r'\node at (6.72,-.05) {提交编号};')
 anns=([(6,'D223',2,.083),(9,r'FA004 5\%',5,.0865),(10,r'FA004 20\%',8,.0928),(11,r'FA004 27.5\%',11,.0878),(15,r'FA004 40\%',14,.0935),(21,'FP01',18,.092),(23,'RH50',23,.092),(35,'K3',28,.090),(36,'FNR',33,.093)] if key=='RankIC' else
       [(7,'QF001',4,-.8),(8,'ST10',8,.7),(9,r'FA004 5\%',5,3.4),(10,r'FA004 20\%',8,5.3),(11,r'FA004 27.5\%',11,4.4),(15,r'FA004 40\%',13,5.5),(21,'FP01',16,6.4),(23,'RH50',23,6.4),(34,'K2',29,2.6),(35,'K3',32,2.8),(36,'FNR',33,6.4)])
 for n,label,tx,ty in anns:
  r=unique[n];x=X(r['plot_x']);y=Y(r[key]);lx=X(tx);ly=Y(ty)
  lines.append(rf'\draw[black!45,line width=.4pt] ({x:.4f},{y:.4f}) -- ({lx:.4f},{ly:.4f});')
  lines.append(rf'\node[fill=white,inner sep=2pt,font=\footnotesize] at ({lx:.4f},{ly:.4f}) {{{label}}};')
lines.append(r'\end{tikzpicture}')
tikz='\n'.join(lines)+'\n'
tex=P/'search_methodology.tex';s=tex.read_text()
s=re.sub(r'% BEGIN SUBMISSION TIKZ.*?% END SUBMISSION TIKZ',lambda _: '% BEGIN SUBMISSION TIKZ\n'+tikz+'% END SUBMISSION TIKZ',s,flags=re.S)
tex.write_text(s)
print(json.dumps(meta,ensure_ascii=False))
