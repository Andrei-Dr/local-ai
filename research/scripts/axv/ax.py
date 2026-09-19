import json,sys,urllib.request,time
def get(u):
    try:
        r=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
        return json.loads(urllib.request.urlopen(r,timeout=30).read())
    except Exception as e: return {'err':str(e)}
for i in sys.argv[1:]:
    d=get("https://api.alphaxiv.org/papers/v3/"+i)
    if 'err' in d: print(i,d);continue
    keys={k:d[k] for k in d if k not in('abstract','citationBibtex','title','authors','topics')}
    print(i,d.get('title','')[:60]); print("  ",{k:v for k,v in keys.items() if k in('metrics','githubUrl','githubStars','resources','commentsCount','visitsCount','publicTotalVotes','citationsCount')} or list(keys)[:30])
    time.sleep(0.5)
