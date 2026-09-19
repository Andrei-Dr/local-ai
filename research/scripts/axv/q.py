import sys,urllib.request,urllib.parse,re,time
import xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom'}
def q(query,n=12,sort='relevance',abs_len=0):
    url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({'search_query':query,'sortBy':sort,'sortOrder':'descending','max_results':n})
    for t in range(4):
        try:
            d=urllib.request.urlopen(url,timeout=60).read();break
        except Exception as e:
            time.sleep(5);d=None
    if not d: print("FAIL",query);return
    r=ET.fromstring(d)
    print("##",query,sort)
    for e in r.findall('a:entry',ns):
        i=e.find('a:id',ns).text.split('/abs/')[-1]
        t=re.sub(r'\s+',' ',e.find('a:title',ns).text)
        p=e.find('a:published',ns).text[:10]
        s=re.sub(r'\s+',' ',e.find('a:summary',ns).text)[:abs_len]
        print(i,p,t); 
        if abs_len: print("   ",s)
    time.sleep(3)
if __name__=="__main__":
    n=int(sys.argv[1]);sort=sys.argv[2];al=int(sys.argv[3])
    for query in sys.argv[4:]: q(query,n,sort,al)
