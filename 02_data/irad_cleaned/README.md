# iRAD crash records: cleaned public release

The police crash reports for Assam (iRAD, Jan 2023 – May 2026), with every personal identifier removed. These are the same 44,114 crashes the model was trained on.

| File | Rows | One row per |
|---|---|---|
| `accidents_clean.csv` | 44,114 | crash: date, hour, district, location (~110 m), severity, casualty counts, road / light / weather conditions |
| `vehicles_clean.csv` | 56,185 | vehicle in a crash: type, class, make (brand only), age, damage, document validity on the crash date |
| `road_details_clean.csv` | 5,730 | road inspection attached to a crash: surface, width, speed limit, markings, suggested remedies |
| `data_dictionary.csv` | 111 | column: file, type, non-null count, description |
| `release_summary.json` | – | row counts, date window, and how many cells each safety check blanked |

The three tables join on `release_id`.

## How it was made

`clean_irad_for_release.py` reads the internal parser output and writes these files. It works as an allow-list: only columns named in the script are published.

**Removed completely:**
- names of drivers, passengers, owners, witnesses, guardians and officers
- phone numbers and addresses
- FIR numbers and police-station addresses
- registration plates
- engine, chassis, insurance-policy, PUC and permit numbers
- source file paths and raw geotag text
- all free text: descriptions, landmarks, road names, damage notes and remarks
- all driver, passenger, pedestrian, witness and hospital tables

**Generalised:**
- The police accident id is replaced by a random `release_id`, so a row can't be traced back to a police case.
- Coordinates are rounded to 3 decimals (about 110 m).
- Time is kept as date + hour only.
- Vehicle make is kept as the brand only (no model, colour or registration date).
- Insurance, PUC, fitness and tax dates are turned into "valid on crash date: yes/no" flags.

**Checked cell by cell:**
1. Text cells that look like a phone number, a plate or an e-mail address are blanked.
2. Rare cells that exactly match an identifier from the internal tables are blanked.
3. VIN-like, trailer-number or date-like values in the vehicle description columns are blanked. The parser sometimes misaligned these columns.

`release_summary.json` lists how many cells each check removed.

**Independent check after cleaning.** We searched every published cell for the real accident ids, FIR numbers, plates, engine and chassis numbers, licence numbers, phone numbers and person names from the internal tables. There were no matches, apart from placeholders such as "UNKNOWN" or "0000000".

## Limits

- Exact crash date + ~110 m location + severity can still point to a crash that was reported in the news. For a stricter release, round coordinates to 2 decimals (about 1 km) or drop `accident_date` and keep only year, month and weekday.
- The `*_valid_on_crash_date` flags come from VAHAN look-ups done at report time, so some may reflect a later status.
- Field reliability varies (see `03_eda`). Behaviour fields are mostly blank.

## Re-running

```
python clean_irad_for_release.py <folder with iRAD_Assam_all__*.parquet> <out folder> <internal folder>
```

The internal folder receives `release_id_mapping_INTERNAL.csv`, which links release ids back to police ids. It must never be published. The raw tables and that mapping stay outside this repository.
