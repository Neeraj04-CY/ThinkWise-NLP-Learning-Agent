import re
s=open(r'c:\ThinkWise\app.py','r',encoding='utf-8').read()
occ = [m.start() for m in re.finditer('"""', s)]
print('count', len(occ))
for i,pos in enumerate(occ[:200]):
    print(i+1,pos, s[pos:pos+80].splitlines()[0])
