p = r'c:\ThinkWise\app.py'
with open(p, 'r', encoding='utf-8') as f:
    lines = f.readlines()
new_lines = [l for l in lines if l.strip() not in ('```', '```python')]
with open(p, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('cleaned')
