tracr-proportion.gv:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-proportion --only-save-canonical --metric=l2

tracr-reverse.gv:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-reverse --only-save-canonical --metric=l2

ioi.gv ioi_heads.gv ioi_heads_qkv.gv:
	python ../../../notebooks/roc_plot_generator.py --task=ioi --only-save-canonical

greaterthan.gv greaterthan_heads.gv greaterthan_heads_qkv.gv:
	python ../../../notebooks/roc_plot_generator.py --task=greaterthan --only-save-canonical

docstring.gv:
	python ../../../notebooks/roc_plot_generator.py --task=docstring --only-save-canonical

%.pdf: %.gv
	neato -Tpdf $< -o $@

all: tracr-proportion.pdf tracr-reverse.pdf ioi.pdf greaterthan.pdf docstring.pdf ioi_heads.pdf ioi_heads_qkv.pdf greaterthan_heads.pdf greaterthan_heads_qkv.pdf

ioi: ioi.pdf ioi_heads.pdf ioi_heads_qkv.pdf

greaterthan: greaterthan.pdf greaterthan_heads.pdf greaterthan_heads_qkv.pdf

tracr: tracr-proportion.pdf tracr-reverse.pdf

docstring: docstring.pdf
