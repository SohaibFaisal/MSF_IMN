file_name = 'elas_cons_comp_ext2.pdf_tex'

with open(file_name, "r") as f:
    lines = f.readlines()

with open(file_name, "w") as f:
    for i,line in enumerate(lines):
        # print(line)
        if 'E1' in line:
            line = line.replace('E1', '$E_x$')
        elif 'E2' in line:
            line = line.replace('E2', '$E_y$')
        elif 'nu12' in line:
            line = line.replace('nu12', '$\\nu_{xy}$')
        elif 'nu23' in line:
            line = line.replace('nu23', '$\\nu_{yz}$')
        elif 'G12' in line:
            line = line.replace('G12', '$G_{xy}$')
        elif 'G23' in line:
            line = line.replace('G23', '$G_{yz}$')

        f.write(line)