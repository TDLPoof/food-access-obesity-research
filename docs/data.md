# Data guide

## Included data

`data/processed/county_obesity_2019_model_data.csv` is the cleaned modeling table copied from the existing project. Each row is a county, identified by a five-character FIPS code. Read FIPS as a string to preserve leading zeros. The repository contains aggregate county statistics, not individual records.

## Sources represented in the preparation code

| Source | Role | Original local input |
| --- | --- | --- |
| CDC PLACES, county data, 2021 release | 2019 age-adjusted adult obesity percentage | `PLACES__Local_Data_for_Better_Health,_County_Data_2021_release_20251101.csv` |
| USDA Food Access Research Atlas | Tract low-access population, aggregated to counties | `FARA/Food Access Research Atlas.csv` |
| USDA Food Environment Atlas | 2016 retail establishments per 1,000 people | `FoodEnvironmentAtlas/StateAndCountyData.csv` |
| Census SAIPE | 2019 poverty percentage | `SAIPE.xls` |
| BLS LAUS | 2019 unemployment percentage | `BLS LAUS 2019 unemployment.xlsx` |
| Census ACS S1501 | High-school and bachelor’s attainment | `CensusEducationData.csv` |
| USDA rural–urban continuum codes | 2013 county classifications | `ruralurbancodes2013.xls` |

The Census education file contains JSON despite its `.csv` extension. Its precise ACS vintage should be verified against the original download before making stronger temporal-alignment claims. Exact original download URLs and source licensing metadata were not preserved in the analysis; the source filenames and checksums in `source_manifest.csv` identify the local inputs used for this analysis.

## Variables

| Column | Meaning |
| --- | --- |
| `fips` | County identifier |
| `obesity_adj_pct` | Age-adjusted adult obesity percentage, prediction target |
| `y_prop` | `obesity_adj_pct / 100`; not population |
| `low_access_share` | Sum of tract `LAPOP1_10` divided by sum of `Pop2010` within county; 1-mile urban / 10-mile rural definition |
| `unhealthy_density` | Fast-food plus convenience outlets per 1,000 people |
| `healthy_density` | Grocery plus specialty food outlets per 1,000 people |
| `swamp_index` | Unhealthy density minus healthy density |
| `swamp_logratio` | `log((unhealthy_density + 0.1) / (healthy_density + 0.1))` |
| `poverty_pct` | Poverty percentage, all ages |
| `hs_plus_pct`, `ba_plus_pct` | High-school-or-higher and bachelor’s-or-higher attainment percentages |
| `unemp_rate` | Unemployment percentage |
| `rucc_2` through `rucc_9` | Rural–urban category indicators; category 1 is the reference |

## Rebuilding from raw sources

Raw source files are not bundled in this repository. The processed dataset supports the full modeling workflow without them. To rebuild from the original local archive, copy the inputs listed above under `data/raw/`, retaining the subdirectories, then run from the repository root:

```bash
python src/build_county_obesity_2019.py
```

For example, when this repository sits beside the original `project` directory, `cp -R ../project/data/raw data/` restores the original inputs. Use the manifest to verify exact source files; current agency releases may use different geography or schemas. The optional `County Populations 2019.csv` fallback is needed only if retail-density columns are absent. The current archived inputs contain those columns.

The script writes the processed CSV and a preparation log. It joins by FIPS and drops incomplete rows. Preserve leading zeros and consult the methods limitations before interpreting excluded counties. Redistribution or reuse should follow each originating agency’s terms.
