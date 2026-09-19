import sys,urllib.request,re,time
import xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
ids=sys.argv[1].split(',')
for k in range(0,len(ids),10):
    url="http://export.arxiv.org/api/query?max_results=20&id_list="+",".join(ids[k:k+10])
    d=None
    for t in range(4):
        try: d=urllib.request.urlopen(url,timeout=60).read();break
        except Exception: time.sleep(5)
    r=ET.fromstring(d)
    for e in r.findall('a:entry',ns):
        i=e.find('a:id',ns).text.split('/abs/')[-1]
        t=re.sub(r'\s+',' ',e.find('a:title',ns).text)
        p=e.find('a:published',ns).text[:10]
        s=re.sub(r'\s+',' ',e.find('a:summary',ns).text)
        c=e.find('x:comment',ns)
        print("=== %s %s %s"%(i,p,t));print(s)
        if c is not None: print("COMMENT:",c.text)
        print()
    time.sleep(3)
