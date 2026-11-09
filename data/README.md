# Where to obtain data

This directory documents the inputs used in the Vartti research project. Obtain any data independently from its provider under the applicable terms. The [dashboard demo](../README.md#try-the-dashboard) and the synthetic model smoke command require no downloads of energy or weather data.

| Source | What to look for |
| --- | --- |
| [Fingrid Open Data](https://data.fingrid.fi/en) | Finnish imbalance prices, mFRR/aFRR, system balance, production, consumption, cross-border flows and day-ahead plans. Start with 165 (consumption forecast), 246 (wind forecast), 247 (solar forecast) and 397 (calculated net imbalance). Register for your own API key if downloading through the API. |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Imbalance prices and load forecasts for SE1, SE3 and EE, and FI–EE transfer capacity. Check the platform's reuse terms for each dataset. |
| [Finnish Meteorological Institute Open Data](https://en.ilmatieteenlaitos.fi/open-data) | Historical station observations, including wind, temperature and precipitation. |
| [eSett Open Data](https://opendata.esett.com/) | Nordic imbalance settlement data. Search for EXP14 for prices and regulation direction. |
| [Nord Pool Power Data Services](https://www.nordpoolgroup.com/en/services/power-market-data-services/) | Finnish and other day-ahead spot prices. Obtain a licence that covers your intended use. The hackathon licence does not carry over. |
| [Nord Pool Urgent Market Messages](https://umm.nordpoolgroup.com/) | Outage and availability disclosures. For historical evaluation, retain each version and its publication time where available. |

For the original challenge specification, contact the organisers via the [Since AI event page](https://sinceai.ai/hackathon). This showcase does not distribute the challenge starter package. Its Nord Pool spot file and the spot-price column in its validation actuals were restricted to the event. Accessing the same prices through a mirror or secondary API does not grant reuse rights.

## Prepare local inputs

1. Download only the series you are authorised to use. Keep source attribution, licence terms, units and time zones with your local copies.
2. Normalise timestamps to UTC and align records to 15-minute delivery periods. Keep the time each record first became usable separate from the delivery time. At a forecast origin, use only the versions published by that origin. Apply the same rule to training labels at the training cutoff.
3. The historical code reads a wide table with one row per delivery period, a column per series and a corresponding `<series>__first_usable_origin_utc` column. See [the input loader and feature definitions](../experiments/imbalance_dl/shared/data.py). The default paths are `data/gold/full.csv` and `data/gold_ext/out/full_ext.parquet`; these paths are for local files. The original builders and scorer are outside the scope of this showcase, so a new dataset may need its own conversion and evaluation code.

Keep downloaded data, derived tables, predictions, API keys and trained weights outside the public repository.
