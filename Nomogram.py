import simple_nomo_modified
import pandas as pd

path = 'Results/nomo_2025.xlsx'
pd.read_excel(path)

cmv_nomo_personalize = simple_nomo_modified.nomogram(path, 
    result_title="Risk", # change the title for the mapping graph
    fig_width=20, # change the width of the figure
    single_height=0.6, # change the height of each axis
    total_point=20,  # chage the maximum point of each variable
    dpi=600,  ### change the resolution
    ax_para = {"c":"black", "linewidth":1.3, "linestyle": "-"}, # change the paramters for each axis
    xtick_para = {"fontsize": 13, "fontfamily": "times new roman"}, # change the parameters for the ticks of each axis, 
    ylabel_para = {"fontsize": 18, "fontfamily": "times new roman", "labelpad":150, 
                  "loc": "center", "color": "black", "rotation":"horizontal"},) # change the parameters for the variable name