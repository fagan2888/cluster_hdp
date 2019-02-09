TARGETS=notes
all: $(TARGETS)

$(TARGETS): %: html/%.html pdf/%.pdf

html/%.html: %.md
	pandoc --self-contained -s -o $@ --highlight-style pygments --data-dir . --columns 1000 --css css/md.css -S --normalize --ascii --number-sections --mathjax="" --mathml $^

pdf/%.pdf: %.md
	pandoc -V lang=fr --listings --latex-engine=xelatex -o $@ --columns 1000 --variable urlcolor=cyan --number-sections $^

clean:
	rm -f pdf/*.pdf html/*.html
