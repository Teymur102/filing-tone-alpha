# Filing Tone Alpha

Does the tone of an earnings release predict returns after it is published? The market prices FinBERT tone within two days (t = 4.58), but there is no tradable drift after that: the out-of-sample long-short portfolio returned −9.2% a year after costs.

![License: MIT](https://img.shields.io/badge/license-MIT-blue) ![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB) ![torch 2.14.1](https://img.shields.io/badge/torch-2.14.1-EE4C2C) ![transformers 5.21.0](https://img.shields.io/badge/transformers-5.21.0-FFD21E) ![statsmodels 0.15.0](https://img.shields.io/badge/statsmodels-0.15.0-4B8BBE) ![pandas 3.1.0](https://img.shields.io/badge/pandas-3.1.0-150458) ![tests 21 passed](https://img.shields.io/badge/tests-21%20passed-brightgreen)

<table><tr>
<td><img src="figures/car_paths_by_tone_tercile.png" alt="CAR paths by tone tercile"></td>
<td><img src="figures/oos_long_short.png" alt="Out-of-sample long-short portfolio"></td>
</tr></table>

## Results

Sample: 2,482 earnings releases (8-K Item 2.02, EX-99.1) from 75 large caps, 2018-01-04 to 2026-07-15.

| Headline | Value |
|---|---|
| FinBERT tone change → CAR[0,+1], per 1 s.d. | +61.1 bp (t = +4.58) |
| LM tone level → CAR[0,+1], per 1 s.d. | +5.4 bp (t = +0.51) |
| FinBERT tone change → CAR[+2,+60], controlling for CAR[0,+1] | −29.4 bp (t = −0.90) |
| OOS 2022-26 long-short drift portfolio, net of 10 bp/side | −9.2% a year (NW t = −2.12, Sharpe −1.00) |
| LM vs FinBERT document tone correlation / same tercile | 0.55 / 54% |

## Proof

- **Tests:** 21 pytest tests cover market-model AR/CAR math, the day-0 market-close rule, LM negation, FinBERT aggregation, the no-look-ahead terciles and portfolio costs. Line coverage is 39%; the untested code is network I/O and plotting.
- **One command:** `make` rebuilds every table and figure. A cold run took about 38 minutes (EDGAR download plus FinBERT on MPS). A cached re-run takes 5 seconds, with a peak memory footprint of 2.4 GB.
- **Data:** [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation) (submissions, filing indexes, XBRL), the [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/), [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) and yfinance prices. No API keys are needed.
- **Deterministic and cached:** every download is saved in `data/` (git-ignored). Re-runs make no network calls and reproduce `results/` exactly.
- **Evaluation:**
  - Market model over [−250,−30]; SEs two-way clustered by firm and quarter.
  - Tercile breakpoints use the prior 365 days only.
  - The signal is chosen on 2018-21 and tested untouched on 2022-26.

## Stack

Python 3.14 · torch 2.14.1 · transformers 5.21.0 · pandas 3.1.0 · numpy 2.5.4 · statsmodels 0.15.0 · yfinance 1.7.0 · matplotlib 3.11.2 · pytest 9.2.0

## Run

```bash
make          # create .venv from requirements.txt, then python run.py
make test     # pytest
```

## Structure

```
run.py                  one-command pipeline entry point
src/tone/               edgar · text · lm · finbert · prices · eventstudy · analysis · figures
tests/                  event-study math, tone aggregation, portfolio
results/                summary.json, tables.md, regression / tercile / portfolio CSVs
figures/                CAR paths, LM vs FinBERT agreement, OOS portfolio
docs/DETAILS.md         full method, results, limitations
```

Full method, limitations and interview-level detail: [docs/DETAILS.md](docs/DETAILS.md) · MIT License

Every number here is produced by the code in this repo.
