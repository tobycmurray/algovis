default: algovis-doc.html
.PHONY: default

%.html: %.ipynb
	jupyter nbconvert --execute --to html $<
