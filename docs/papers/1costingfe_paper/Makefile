PAPER = 1costingfe_paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all paper rebuild clean

all: $(PAPER).pdf

# Convenience alias: `make paper`
paper: $(PAPER).pdf

# Force a full rebuild even if the PDF looks up to date
rebuild:
	touch $(PAPER).tex
	$(MAKE) paper

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	bibtex $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc
