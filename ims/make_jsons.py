#%%

import plotly

# Set renderers 
plotly.io.renderers.default = "jupyterlab" # apparently the secret sauce to make latex work in notebooks (not requireing setting renderer to browser?)
# that failed 

plotly.io.renderers.default = "browser"

# note: LATEX works in scripts, 


# %%

fname = "induction_json.json"
plotly_graph = plotly.io.read_json(fname)

# %%

plotly_graph.show()

# %%

# Get new data 
paper_corrupted_fname = "current_paper_induction_json.json"
paper_corrupted_figure = plotly.io.read_json(paper_corrupted_fname)
paper_zero_figure = plotly.io.read_json("current_paper_induction_zero.json")

# %%

scatter_names = [
    "ACDC", 
    "SP", 
    "HISP",
]

x_data = {}
y_data = {}
color_data = {}

figures = {
    "corrupted": paper_corrupted_figure,
    "zero": paper_zero_figure,
}

for figure_name, figure in figures.items():
    for name in scatter_names:
        x_data[name] = []
        y_data[name] = []

        x_data_element = [thing for thing in figure["data"] if thing["name"] == name][0]
        y_data_element = [thing for thing in figure["data"] if thing["name"] == name][0]

        x_data[(figure_name, name)] = x_data_element["x"]
        y_data[(figure_name, name)] = y_data_element["y"]

        color_data[(figure_name, name)] = x_data_element["marker"]["color"]

# %%

# Add this into the paper figure

the_ref = {}

for i in range(len(plotly_graph.data)):
    if plotly_graph.data[i]["yaxis"] == "y" and plotly_graph.data[i]["name"] in scatter_names:
        plotly_graph.data[i]["x"] = x_data[("corrupted", plotly_graph.data[i]["name"])]
        plotly_graph.data[i]["y"] = y_data[("corrupted", plotly_graph.data[i]["name"])]
        if plotly_graph.data[i]["name"] == "ACDC":
            cur_col_data = color_data[("corrupted", plotly_graph.data[i]["name"])]
            plotly_graph.data[i]["marker"]["color"] = cur_col_data
        the_ref[plotly_graph.data[i]["name"]] =plotly_graph.data[i]["marker"]

    if plotly_graph.data[i]["yaxis"] == "y2" and plotly_graph.data[i]["name"] in scatter_names:
        plotly_graph.data[i]["x"] = x_data[("zero", plotly_graph.data[i]["name"])]
        plotly_graph.data[i]["y"] = y_data[("zero", plotly_graph.data[i]["name"])]
        # if plotly_graph.data[i]["name"] == "ACDC":
        #     cur_col_data = color_data[("zero", plotly_graph.data[i]["name"])]
        #     plotly_graph.data[i]["marker"]["color"] = cur_col_data
            # cur_col_data = color_data[("corrupted", plotly_graph.data[i]["name"])]


for i in range(len(plotly_graph.data)):
    if plotly_graph.data[i]["yaxis"] == "y2" and plotly_graph.data[i]["name"] in scatter_names:
        print(plotly_graph.data[i])
        print("Hey")
        plotly_graph.data[i]["marker"] = the_ref[plotly_graph.data[i]["name"]] # color_data[("corrupted", plotly_graph.data[i]["name"])]

# %%

plotly_graph.show()

# %%

lis = []

for i in range(len(plotly_graph.data)):
    if plotly_graph.data[i]["yaxis"] == "y3":
        # print(plotly_graph.data[i])
        # break        
        # print(i)
        lis.append(i)

# %%

# Remove this from the paper figure

for l in sorted(lis[1:], reverse=True):
    plotly_graph.data = plotly_graph.data[:l] + plotly_graph.data[l+1:]

# %%

fig = plotly.io.read_json("my_new_plotly_graph.json")

# %%

fig.show()
# %%

# TODO add better legend
