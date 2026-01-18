# Development Log – AI Duplicate Image Checker

## Week 1 – Problem identification
- Identified repeated images during amends as a manual pain point
- Explored whether Pages documents could expose image usage
- Decided to work via PDF export for reliability

## Week 2 – Initial prototype
- Built CLI tool to extract image regions from PDFs
- Implemented AI embeddings to detect visual similarity
- Identified false positives caused by icons and templates

## Week 3 – Accuracy improvements
- Added size and aspect-ratio filtering
- Tuned similarity thresholds
- Grouped results by image clusters with page numbers

## Week 4 – Desktop application
- Built Tkinter UI to remove command-line dependency
- Added image previews and result table
- Sorted output by most duplicated images first

## Week 5 – Accessibility & branding
- Reworked UI to a white, high-contrast design
- Used FS Me font for accessibility consistency
- Applied Easy Read Online brand colours for highlights

## Week 6 – Workflow integration
- Added drag & drop support
- Added Pages document support via automatic PDF export
- Created Automator launcher for non-technical users

## Reflection
This project strengthened my understanding of:
- Applied AI in real workflows
- Accessible UI design
- Translating technical tools into practical workplace solutions
