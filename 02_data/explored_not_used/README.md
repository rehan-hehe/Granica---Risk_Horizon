# Explored, not used in the model

## CE323 field studies (`ce323_field_studies/`)

**What:** hand-collected traffic studies from IIT Guwahati's CE323 lab, digitised into 23 tables:
- classified volume counts and headways
- spot speeds and moving-observer runs
- turning movements
- parking accumulation
- origin–destination interviews

**How:** `build.py` and `data_manual.py` type the report values into tables. `verify.py` runs 21 consistency checks (all pass). `style.py` formats the workbook.

**Problems found** (see `Validation_Log.csv`):
- edited video timestamps implying **2,571 km/h**
- swapped free/combined labels
- impossible percentiles

**Why it's not in the model:**
- iRAD crashes cover all of Assam over 41 months; these counts cover a few Guwahati sites on single days, so they can't be joined to crash segments at scale.
- They are the ground truth we need for the next step: converting live TomTom speeds into traffic exposure.

We kept them for that step. Data is at group level; licence plates are pseudonymised.
