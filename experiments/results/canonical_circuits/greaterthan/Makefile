layout.\#%.gv: \#%.gv
	neato -olayout.$@ $<

layout.gv: $(wildcard layout.*.gv)
	gvpack -Glayout=neato -Goverlap=false -Gsplines=true -Gbgcolor=transparent -o$@ $^
