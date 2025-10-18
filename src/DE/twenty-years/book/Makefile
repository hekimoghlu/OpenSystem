SOURCES =					\
	applications.tex			\
	desktop-shell.tex			\
	eazel.tex				\
	enemies.tex				\
	glossary-entries.tex			\
	graphics.tex				\
	hardware.tex				\
	historybit.tex				\
	intro.tex				\
	languages.tex				\
	main.tex				\
	other-subsystems.tex			\
	pants.tex				\
	prehistory.tex				\
	projects-faded.tex			\
	references.bib				\
	rhad-labs.tex				\
	swedish.tex				\
	text.tex				\
	ximian.tex				\
	yorba.tex

BIB_GENERATED = references_bibertool.bib

main.pdf: $(SOURCES) $(BIB_GENERATED)
	xelatex main
	biber main.bcf
	makeglossaries main
	makeindex main
	xelatex main

references_bibertool.bib: references.bib $(SOURCES)
	biber --tool references.bib
