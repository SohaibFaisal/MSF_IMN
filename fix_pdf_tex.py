file_name = 'mesh_ind_combined_lc2.pdf_tex'



ff = [f'res{x}{y}.pdf_tex' for x in range(1,9) for y in ['a', 'b', 'c', 'd']]
print(ff)

for file_name in ff:
    with open(file_name, "r") as f:
        lines = f.readlines()


    ##########################################################################

    tension = True
    with open(file_name, "w") as f:
        for i,line in enumerate(lines):
            if 'a' in file_name or 'c' in file_name:
                if 'Stress' in line:
                    line = line.replace(r'Stress ($\sigma_y$) ($\sigma_y$) [MPa],', 'Stress ($\\sigma_y$)')
                elif 'Strain' in line:
                    line = line.replace(r'Strain ($\epsilon_y$)  ($\epsilon_y$) [-],', 'Strain ($\\epsilon_y$) ')

            elif 'b' in file_name or 'd' in file_name:
                if 'Stress' in line:
                    line = line.replace(r'Stress ($\tau_{yz}$) ($\tau_{yz}$) [MPa],', 'Stress ($\\tau_{yz}$)')
                elif 'Strain' in line:
                    line = line.replace(r'Strain ($\gamma_{yz}$) ($\gamma_{yz}$) [-],', 'Strain ($\\gamma_{yz}$)')


            f.write(line)