# IBM Applied Data Science Capstone — SpaceX Falcon 9

**Analytical presentation:** M. Nicholas de Boulet · 9 October 2026

## Project objective
Study historical Falcon 9 first-stage landing outcomes using a reproducible examination of publicly shared IBM-course-style data, descriptive statistics, SQL, geographical comparisons, and classification benchmarks.

## Scope and disclosure
The report contains an **independent reconstruction** based on a third-party public educational reference archive. The learner completed graded course assessments, **not the associated original hands-on project notebooks**. This repository does not claim otherwise. Do not represent the supplied third-party notebooks as original work.

## Data sources
- `dataset_part_2.csv`: 90-row exploratory and predictive-analysis cohort
- `spacex_launch_geo.csv`: 56-row geographic study cohort
- `spacex_launch_dash.csv`: dashboard cohort (56 records)
- `my_data1.db`: course-style archived SQLite data

The datasets differ in coverage and must not be assumed to represent identical samples.

## Reproducibility
`independent_reconstruction_reference.py` documents the analysis that was independently run to prepare the 47-slide presentation. The script was designed for the local authoring environment, with absolute paths and a dependency on the IBM PPTX template. It is **not** a ready-to-run standalone file: adjust its paths and dependencies before execution.

Model evaluation used an 80/20 stratified train/test split, 5-fold cross-validation on the training fold, and an untouched holdout evaluation. The historic sample is not suitable for deployed landing predictions.

## Source attribution
IBM Skills Network — Applied Data Science Capstone, “Winning Space Race with Data Science” presentation template.
Reference datasets originated from the publicly shared educational repository `Ahmed-Nezar/IBM-Data-Science-Capstone-SpaceX`; analyze licensing and permitted reuse before redistribution.

## Submitted presentation
`Data_Science_Capstone_Project_Report_FINAL.pdf` should be uploaded separately after author review; include it in the repository if permitted.
