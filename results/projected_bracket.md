# Projected World Cup 2026 Bracket — with head-to-head matchup odds

A single 'most likely' bracket built by taking the most probable occupant of every slot from the simulation's per-team probabilities, then computing the *exact* head-to-head advancement odds for each resulting matchup directly from the fitted Dixon-Coles model (regulation -> extra time -> penalties, in closed form -- not re-sampled).

Strength basis: `data/raw/elo_ratings.csv (Elo + host bump approximation)`

> **Read this as one plausible storyline, not a forecast of the joint outcome.** Both the projected group results and the bracket built on top of them are point estimates from marginal probabilities -- real tournaments branch at every match.

## Projected group-stage results

| Group | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| A | Mexico | South Korea | Czechia | South Africa |
| B | Canada | Switzerland | Bosnia and Herzegovina | Qatar |
| C | Brazil | Morocco | Scotland | Haiti |
| D | Türkiye | Australia | Paraguay | United States |
| E | Ecuador | Germany | Côte d'Ivoire | Curaçao |
| F | Netherlands | Japan | Sweden | Tunisia |
| G | Belgium | Iran | Egypt | New Zealand |
| H | Spain | Uruguay | Saudi Arabia | Cabo Verde |
| I | France | Norway | Senegal | Iraq |
| J | Argentina | Austria | Algeria | Jordan |
| K | Colombia | Portugal | Uzbekistan | DR Congo |
| L | England | Croatia | Panama | Ghana |

**Best third-place qualifiers projected from groups:** A, B, C, E, F, I, K, L

## Round of 32

| Matchup | Advance odds | Favorite | Path to a draw |
|---|---|---|---|
| **Ecuador** (E) vs **Czechia** (A) | Ecuador 75.7% — 24.3% Czechia | **Ecuador** | P(level after 90)=26.7%, P(to penalties)=14.4%, P(Ecuador wins pens)=60.0% |
| **France** (I) vs **Scotland** (C) | France 81.9% — 18.1% Scotland | **France** | P(level after 90)=23.1%, P(to penalties)=11.9%, P(France wins pens)=60.0% |
| **South Korea** (A) vs **Switzerland** (B) | South Korea 40.5% — 59.5% Switzerland | **Switzerland** | P(level after 90)=31.3%, P(to penalties)=17.8%, P(South Korea wins pens)=46.2% |
| **Netherlands** (F) vs **Morocco** (C) | Netherlands 52.5% — 47.5% Morocco | **Netherlands** | P(level after 90)=32.0%, P(to penalties)=18.3%, P(Netherlands wins pens)=51.0% |
| **Portugal** (K) vs **Croatia** (L) | Portugal 59.0% — 41.0% Croatia | **Portugal** | P(level after 90)=31.4%, P(to penalties)=17.9%, P(Portugal wins pens)=53.6% |
| **Spain** (H) vs **Austria** (J) | Spain 84.4% — 15.6% Austria | **Spain** | P(level after 90)=21.4%, P(to penalties)=10.7%, P(Spain wins pens)=60.0% |
| **Türkiye** (D) vs **Bosnia and Herzegovina** (B) | Türkiye 83.7% — 16.3% Bosnia and Herzegovina | **Türkiye** | P(level after 90)=21.9%, P(to penalties)=11.1%, P(Türkiye wins pens)=60.0% |
| **Belgium** (G) vs **Côte d'Ivoire** (E) | Belgium 66.5% — 33.5% Côte d'Ivoire | **Belgium** | P(level after 90)=29.9%, P(to penalties)=16.8%, P(Belgium wins pens)=56.8% |
| **Brazil** (C) vs **Japan** (F) | Brazil 58.9% — 41.1% Japan | **Brazil** | P(level after 90)=31.4%, P(to penalties)=17.9%, P(Brazil wins pens)=53.6% |
| **Germany** (E) vs **Norway** (I) | Germany 53.2% — 46.8% Norway | **Germany** | P(level after 90)=32.0%, P(to penalties)=18.3%, P(Germany wins pens)=51.3% |
| **Mexico** (A) vs **Sweden** (F) | Mexico 81.1% — 18.9% Sweden | **Mexico** | P(level after 90)=23.6%, P(to penalties)=12.2%, P(Mexico wins pens)=60.0% |
| **England** (L) vs **Uzbekistan** (K) | England 78.6% — 21.4% Uzbekistan | **England** | P(level after 90)=25.1%, P(to penalties)=13.3%, P(England wins pens)=60.0% |
| **Argentina** (J) vs **Uruguay** (H) | Argentina 74.7% — 25.3% Uruguay | **Argentina** | P(level after 90)=27.1%, P(to penalties)=14.7%, P(Argentina wins pens)=60.0% |
| **Australia** (D) vs **Iran** (G) | Australia 50.0% — 50.0% Iran | Pick'em | P(level after 90)=32.0%, P(to penalties)=18.3%, P(Australia wins pens)=50.0% |
| **Canada** (B) vs **Senegal** (I) | Canada 56.7% — 43.3% Senegal | **Canada** | P(level after 90)=31.7%, P(to penalties)=18.1%, P(Canada wins pens)=52.7% |
| **Colombia** (K) vs **Panama** (L) | Colombia 73.5% — 26.5% Panama | **Colombia** | P(level after 90)=27.6%, P(to penalties)=15.1%, P(Colombia wins pens)=60.0% |

## Round of 16

| Matchup | Advance odds | Favorite | Path to a draw |
|---|---|---|---|
| **Ecuador** (E) vs **France** (I) | Ecuador 37.7% — 62.3% France | **France** | P(level after 90)=30.9%, P(to penalties)=17.5%, P(Ecuador wins pens)=45.0% |
| **Switzerland** (B) vs **Netherlands** (F) | Switzerland 44.0% — 56.0% Netherlands | **Netherlands** | P(level after 90)=31.8%, P(to penalties)=18.1%, P(Switzerland wins pens)=47.6% |
| **Portugal** (K) vs **Spain** (H) | Portugal 28.9% — 71.1% Spain | **Spain** | P(level after 90)=28.5%, P(to penalties)=15.7%, P(Portugal wins pens)=41.2% |
| **Türkiye** (D) vs **Belgium** (G) | Türkiye 51.6% — 48.4% Belgium | **Türkiye** | P(level after 90)=32.0%, P(to penalties)=18.3%, P(Türkiye wins pens)=50.6% |
| **Brazil** (C) vs **Germany** (E) | Brazil 58.5% — 41.5% Germany | **Brazil** | P(level after 90)=31.5%, P(to penalties)=17.9%, P(Brazil wins pens)=53.4% |
| **Mexico** (A) vs **England** (L) | Mexico 46.4% — 53.6% England | **England** | P(level after 90)=31.9%, P(to penalties)=18.3%, P(Mexico wins pens)=48.6% |
| **Argentina** (J) vs **Iran** (G) | Argentina 81.0% — 19.0% Iran | **Argentina** | P(level after 90)=23.7%, P(to penalties)=12.3%, P(Argentina wins pens)=60.0% |
| **Canada** (B) vs **Colombia** (K) | Canada 39.5% — 60.5% Colombia | **Colombia** | P(level after 90)=31.2%, P(to penalties)=17.7%, P(Canada wins pens)=45.7% |

## Quarterfinals

| Matchup | Advance odds | Favorite | Path to a draw |
|---|---|---|---|
| **France** (I) vs **Netherlands** (F) | France 64.5% — 35.5% Netherlands | **France** | P(level after 90)=30.4%, P(to penalties)=17.1%, P(France wins pens)=55.9% |
| **Spain** (H) vs **Türkiye** (D) | Spain 78.5% — 21.5% Türkiye | **Spain** | P(level after 90)=25.2%, P(to penalties)=13.3%, P(Spain wins pens)=60.0% |
| **Brazil** (C) vs **England** (L) | Brazil 47.7% — 52.3% England | **England** | P(level after 90)=32.0%, P(to penalties)=18.3%, P(Brazil wins pens)=49.1% |
| **Argentina** (J) vs **Colombia** (K) | Argentina 65.6% — 34.4% Colombia | **Argentina** | P(level after 90)=30.2%, P(to penalties)=16.9%, P(Argentina wins pens)=56.4% |

## Semifinals

| Matchup | Advance odds | Favorite | Path to a draw |
|---|---|---|---|
| **France** (I) vs **Spain** (H) | France 38.5% — 61.5% Spain | **Spain** | P(level after 90)=31.0%, P(to penalties)=17.6%, P(France wins pens)=45.4% |
| **England** (L) vs **Argentina** (J) | England 37.3% — 62.7% Argentina | **Argentina** | P(level after 90)=30.8%, P(to penalties)=17.4%, P(England wins pens)=44.9% |

## Final

| Matchup | Advance odds | Favorite | Path to a draw |
|---|---|---|---|
| **Spain** (H) vs **Argentina** (J) | Spain 53.7% — 46.3% Argentina | **Spain** | P(level after 90)=31.9%, P(to penalties)=18.3%, P(Spain wins pens)=51.5% |

## Projected champion: Spain

*Methodology: matchup odds are the model's exact analytical probabilities for that specific pairing -- P(advance) = P(win in regulation) + P(level after 90) x [P(win in ET) + P(level after 120) x P(win on penalties)] -- evaluated from the same joint Dixon-Coles scoreline distribution `simulate_match` samples from. They are not empirical counts of how often these two sides happened to meet across simulated tournaments, so they carry no Monte Carlo sampling noise.*