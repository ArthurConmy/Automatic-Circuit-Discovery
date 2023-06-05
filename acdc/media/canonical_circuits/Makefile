tracr-proportion.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-proportion --only-save-canonical --metric=l2

tracr-reverse.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-reverse --only-save-canonical --metric=l2

ioi.pdf ioi.gv ioi_heads.gv ioi_heads_qkv.gv:
	python ../../../notebooks/roc_plot_generator.py --task=ioi --only-save-canonical

ioi_heads.pdf: ioi_heads.gv
	dot -Tpdf ioi_heads.gv -o ioi_heads.pdf

ioi_heads_qkv.pdf: ioi_heads_qkv.gv
	dot -Tpdf ioi_heads_qkv.gv -o ioi_heads_qkv.pdf

greaterthan.pdf greaterthan.gv greaterthan_heads.gv greaterthan_heads_qkv.gv:
	python ../../../notebooks/roc_plot_generator.py --task=greaterthan --only-save-canonical

greaterthan_heads.pdf: greaterthan_heads.gv
	dot -Tpdf greaterthan_heads.gv -o greaterthan_heads.pdf

greaterthan_heads_qkv.pdf: greaterthan_heads_qkv.gv
	dot -Tpdf greaterthan_heads_qkv.gv -o greaterthan_heads_qkv.pdf

docstring.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=docstring --only-save-canonical


all: tracr-proportion.pdf tracr-reverse.pdf ioi.pdf greaterthan.pdf docstring.pdf ioi_heads.pdf ioi_heads_qkv.pdf greaterthan_heads.pdf greaterthan_heads_qkv.pdf
