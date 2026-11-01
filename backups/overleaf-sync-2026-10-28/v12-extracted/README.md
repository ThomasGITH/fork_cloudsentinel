# Master Thesis Template for Master Software Engineering @ UvA

This is the template to use for your master thesis Software Engineering. It provides you with a structure that we expect you to follow, with guidelines on the contents of each section. Depending on your research you can deviate from this structure, usually by renaming or adding some chapters in the middle. Do discuss this with your academic supervisor first!

Be sure to change the variables under the "Options" banner in `main.tex`, such as title, subtitle, author, examiner, company, etc., and pick a cover image (or remove it entirely).

## Layout

The files and directories in this project are layed out as follows:

- `main.tex` — top-level LaTeX file that assembles the thesis (loads options, cover, front matter and chapters).
- `references.bib` - bibliography file for citation management
- `acronyms.tex` - collection of acrynyms and their expansions
- `chapters/` — individual chapter source files (e.g. `introduction.tex`, `related_work.tex`) intended to be included from `main.tex`.
- `frontmatter/` — files for title page, abstract, acknowledgements, and other pre-chapter material.
- `appendix/` — appendices (e.g. `appendix-a.tex`).
- `figures/` — all graphics (PDF, PNG, SVG) used in the thesis.
- `build/` — generated output (PDF, intermediate files). Ignored by Git via `.gitignore`.
- `Makefile`, `latexmkrc.local` — build helpers for local building.
- `README.md` - this file.

## Local Build Requirements

 * Latest [TeX Live](https://www.tug.org/texlive/) (2026 at the time of writing) or [MacTex](http://www.tug.org/mactex/) (2026, based on TeX Live)

 The following requirements are part of TeX Live:
  * [`latexmk`](http://www.ctan.org/pkg/latexmk/) (version 4.88)
  * [LuaLaTex](http://luatex.org/) (LuaHBTeX version 1.24.0)

## Build Instructions

To build the thesis, cd to the root of the directory, and use the supplied Makefile:

    $ make

Clean up all non-essential files (except pdf):

    $ make clean

Clean up all non-essential files (including pdf):

    $ make cleanall

Continuously build on any changes and update the pdf viewer:

    $ make watch

Set the default pdfviewer via `~/.latexmkrc`, for example for Evince:

    $pdf_previewer = "start evince %O %S";
    $pdf_update_method = 0;

For [MuPDF](https://mupdf.com):

    $pdf_previewer = "start mupdf %O %S";
    $pdf_update_method = 2;
    $dvi_update_signal = SIGHUP;

For [Skim](http://skim-app.sourceforge.net):

	$pdf_previewer = "open -a Skim"

Create an export copy of the generated pdf, based on the current Git revision id:

    $ make export

Get a wordcount of all TeX files:

    $ make wordcount
