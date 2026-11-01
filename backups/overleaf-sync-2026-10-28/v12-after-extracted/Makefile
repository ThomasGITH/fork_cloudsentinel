# Latex Makefile using latexmk
# Downloaded from https://gist.github.com/dogukancagatay/2eb82b0233829067aca6
# Originally from http://tex.stackexchange.com/a/40759

# Main tex file
PROJNAME := main
PROJDIR := .

LATEXMK_CMD := latexmk -r latexmkrc.local -cd

.PHONY: $(PROJNAME).pdf all export wordcount clean watch cleanall

all: $(PROJNAME).pdf

# CUSTOM BUILD RULES
# In case you didn't know, '$@' is a variable holding the name of the target,
# and '$<' is a variable holding the (first) dependency of a rule.
# "raw2tex" and "dat2tex" are just placeholders for whatever custom steps
# you might have.

# %.tex: %.raw
# 	./raw2tex $< > $@
# 
# %.tex: %.dat
# 	./dat2tex $< > $@

# MAIN LATEXMK RULE

$(PROJNAME).pdf: $(PROJDIR)/$(PROJNAME).tex
	$(LATEXMK_CMD) -time -pdflua -use-make $<

watch: $(PROJDIR)/$(PROJNAME).tex
	$(LATEXMK_CMD) -silent -time -pdflua -pvc -use-make $<
	
export: $(PROJNAME).pdf
	rm -f build/$(PROJNAME)-???????.pdf && cp build/$< build/$(PROJNAME)-$(shell git rev-parse --short HEAD).pdf
	
wordcount:
	@find $(PROJDIR) -name '*.tex' | xargs texcount -total -nosub -sum 2>/dev/null

cleanall:
	$(LATEXMK_CMD) -C $(PROJDIR)/$(PROJNAME).tex

clean:
	$(LATEXMK_CMD) -c $(PROJDIR)/$(PROJNAME).tex

