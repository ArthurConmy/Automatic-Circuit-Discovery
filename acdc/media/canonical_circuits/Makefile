tracr-proportion.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-proportion --only-save-canonical --metric=l2

tracr-reverse.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=tracr-reverse --only-save-canonical --metric=l2

ioi.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=ioi --only-save-canonical

greaterthan.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=greaterthan --only-save-canonical

docstring.pdf:
	python ../../../notebooks/roc_plot_generator.py --task=docstring --only-save-canonical


all: tracr-proportion.pdf tracr-reverse.pdf ioi.pdf greaterthan.pdf docstring.pdf
