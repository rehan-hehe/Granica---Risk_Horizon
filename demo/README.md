# Demo · 48-hour field record

Open `index.html` in a browser (keep `data2.js` next to it). Use the arrow keys to move between the 8 screens; press space on screen 4 to replay the live collection.

| Screen | Shows |
|---|---|
| 0 Question | the problem, the users, and our 48-hour path |
| 1 Police records | how usable each iRAD field is, the three checks that changed the plan, and crashes per month (reporting drift) |
| 2 Gaps → sources | each gap in the records, the source we went to, and what we found, including the CE323 lab data we explored but could not use |
| 3 Road network | the Guwahati corridor from OSM with observed serious crashes, TomTom tiles and hotspot points |
| 4 Live traffic | TomTom polls every 15 minutes, a speed heatmap and the collection log (manifest.jsonl) |
| 5 Features | 52 → 43 → 33 features, source ablation, and the leak we caught |
| 6 Preliminary check | basic count models tested once on Jul 2025 – May 2026 |
| 7 Next | what the data cannot tell yet, and the next collection |

`data2.js` holds only aggregates: road geometry, crash counts per road piece (capped at 9), and TomTom summaries. It contains no crash-level records or personal data.
