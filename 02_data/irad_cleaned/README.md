# iRAD crash records (cleaned release)

Upload the cleaned tables here as CSV. Use `clean_irad_for_release.py` as a starting point.

**Remove:**
- names, phone numbers, addresses, father/guardian names
- licence, registration, engine and chassis numbers
- ID documents and free-text narratives that may name people

**Keep:**
- accident id
- date and time
- latitude and longitude
- district
- severity, killed and injured counts
- collision type
- road, light and weather conditions
- vehicle types

**Before uploading,** check the output manually. Automatic column matching is not a guarantee.

**Recommended files:** `accidents_clean.csv`, `vehicles_clean.csv` (types only), `road_details_clean.csv`, `data_dictionary.csv`.
