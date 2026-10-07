with open('for_figs.txt', 'r') as f:
    lines = f.readlines()
    figs = []
    for l in lines:
        if '\\import{Figures/}' in l:
            if '%' not in l:
                figs.append(l.replace('\\import{Figures/}', '').strip().replace('{','').replace('}',''))

figs.sort()
for f in figs:
    print(f)