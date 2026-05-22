# SQL Transfer-Radius Diagnostic

Input gate: `/Users/rishisim/Documents/research/ltm-research/intercode_sql_runs/gates/gpt-5.4/sql_action_stable_50`

## Final Metrics
| framework | success_total | accuracy | avg_reward | avg_steps |
| --- | --- | --- | --- | --- |
| ReAct | 43 / 50 | 0.8600 | 0.8999 | 7.4800 |
| CR | 41 / 50 | 0.8200 | 0.8688 | 5.5600 |
| TR | 42 / 50 | 0.8400 | 0.8911 | 5.8200 |
| CR+TR | 39 / 50 | 0.7800 | 0.8361 | 5.7200 |
| hard-neg CR+TR | 41 / 50 | 0.8200 | 0.8811 | 5.9400 |

## Delta Status Summary
| variant | regressions | reward_drops | improvements | reward_gains | changed_final_sql | help_calls |
| --- | --- | --- | --- | --- | --- | --- |
| CR | 3 | 0 | 1 | 0 | 41 | 0 |
| TR | 1 | 0 | 0 | 0 | 41 | 0 |
| CR+TR | 4 | 0 | 0 | 0 | 39 | 0 |
| hard-neg CR+TR | 2 | 0 | 0 | 0 | 40 | 0 |

## Failure Mode Summary
| variant | failure_mode | count | tasks |
| --- | --- | --- | --- |
| CR | join inclusivity / zero-row inclusion | 1 | sql_66 |
| CR | set granularity / duplicate collapse | 1 | sql_120 |
| CR | spurious temporal normalization | 1 | sql_49 |
| TR | prompt-only join inclusivity drift | 1 | sql_66 |
| CR+TR | join inclusivity / zero-row inclusion | 2 | sql_66, sql_117 |
| CR+TR | output column drift | 1 | sql_185 |
| CR+TR | set granularity / duplicate collapse | 1 | sql_120 |
| hard-neg CR+TR | duplicate policy drift | 1 | sql_38 |
| hard-neg CR+TR | prompt-only join inclusivity drift | 1 | sql_66 |

## Paper-Ready Diagnostic Summary
The trusted SQL gate isolates memory behavior rather than runner instability: the paired run has zero retrieved-help calls across memory variants and no environment-error matches in world logs. TR therefore measures a prompt/tool-availability perturbation, not useful tool retrieval.

SQL looks syntactically regular, but the transfer radius of prior memories is narrow. A memory that is directionally plausible often changes a latent query policy: whether to include zero-count entities, whether duplicate rows are meaningful, whether exclusion applies to rows or grouped entities, which columns the grader expects, or whether literal text dates should be normalized. These choices are small in syntax and large in denotation, so near-miss memories can degrade exact-match reward.

The clearest CR regressions are near-miss transfers. In `sql_66`, memories about empty tables and count verification push the agent toward a left join that adds stadiums with zero concerts. In `sql_120`, a country-level memory about 'not speaking English' transfers to channel rows and collapses valid duplicate `(aspect ratio, country)` rows with DISTINCT. In `sql_185`, CR+TR adds a plausible explanatory `note` column even though the task asks only for death and injury values. In `sql_117`, counting a specific joined id is reasonable under an inner join, but paired with a left join it admits airlines with no flights.

Hard-negative CR+TR performs relatively well because irrelevant memories are easier to ignore than semantically adjacent memories. Its remaining failures are mostly generic policy drift: `sql_38` drops DISTINCT and duplicates an otherwise correct row, while `sql_66` repeats the same zero-row inclusion error seen under TR. The contrast supports the paper's claim that SQL transfer fails not because SQL lacks structure, but because the structure creates many locally plausible, task-specific policy choices with fragile boundaries.

## Largest Regressions vs ReAct
| variant | task_id | task_desc | delta_reward | status | failure_mode | react_final_sql | variant_final_sql |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | sql_49 | list all cartoon titles and their directors ordered by their air date | -1.0300 | regression | spurious temporal normalization | SELECT Title, Directed_by FROM cartoon ORDER BY Original_air_date | SELECT Title, Directed_by FROM cartoon ORDER BY STR_TO_DATE(Original_air_date, '%M%e,%Y'); |
| CR+TR | sql_117 | Which airlines have less than 200 flights? | -1.0000 | regression | join inclusivity / zero-row inclusion | SELECT a.Airline FROM airlines AS a JOIN flights AS f ON a.uid = f.Airline GROUP BY a.uid, a.Airline HAVING COUNT(*) < 200 | SELECT a.Airline FROM airlines AS a LEFT JOIN flights AS f ON a.uid = f.Airline GROUP BY a.uid, a.Airline HAVING COUNT(f.Airline) < 200; |
| CR+TR | sql_185 | What are the death and injury situations caused by the ship with tonnage 't'? | -1.0000 | regression | output column drift | SELECT d.killed, d.injured FROM death AS d JOIN ship AS s ON d.caused_by_ship_id = s.id WHERE s.tonnage = 't'; | SELECT d.note, d.killed, d.injured FROM death AS d JOIN ship AS s ON d.caused_by_ship_id = s.id WHERE s.tonnage = 't'; |
| CR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | -0.7500 | regression | set granularity / duplicate collapse | SELECT Pixel_aspect_ratio_PAR, Country FROM tv_channel WHERE Language != 'English' | SELECT DISTINCT Pixel_aspect_ratio_PAR, Country FROM tv_channel WHERE id NOT IN ( SELECT id FROM tv_channel WHERE Language = 'English' ); |
| CR+TR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | -0.7500 | regression | set granularity / duplicate collapse | SELECT Pixel_aspect_ratio_PAR, Country FROM tv_channel WHERE Language != 'English' | SELECT DISTINCT Pixel_aspect_ratio_PAR, Country FROM tv_channel WHERE Language <> 'English'; |
| hard-neg CR+TR | sql_38 | List the name and date the battle that has lost the ship named 'Lettice' and the ship named 'HMS Atalanta' | -0.5000 | regression | duplicate policy drift | SELECT DISTINCT b.name, b.date FROM ship AS s JOIN battle AS b ON s.lost_in_battle = b.id WHERE s.name IN ('Lettice', 'HMS Atalanta'); | SELECT b.name, b.date FROM ship AS s JOIN battle AS b ON s.lost_in_battle = b.id WHERE s.name IN ('Lettice', 'HMS Atalanta'); |
| CR | sql_66 | Show the stadium name and the number of concerts in each stadium. | -0.4400 | regression | join inclusivity / zero-row inclusion | SELECT s.Name, COUNT(c.concert_ID) AS num_concerts FROM stadium AS s JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Name | SELECT s.Name, COUNT(c.concert_ID) AS number_of_concerts FROM stadium AS s LEFT JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID, s.Name; |
| CR+TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | -0.4400 | regression | join inclusivity / zero-row inclusion | SELECT s.Name, COUNT(c.concert_ID) AS num_concerts FROM stadium AS s JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Name | SELECT s.Name, COUNT(c.concert_ID) AS number_of_concerts FROM stadium AS s LEFT JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID, s.Name; |
| TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | -0.4400 | regression | prompt-only join inclusivity drift | SELECT s.Name, COUNT(c.concert_ID) AS num_concerts FROM stadium AS s JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Name | SELECT s.Name, COUNT(c.concert_ID) AS number_of_concerts FROM stadium AS s LEFT JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID, s.Name; |
| hard-neg CR+TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | -0.4400 | regression | prompt-only join inclusivity drift | SELECT s.Name, COUNT(c.concert_ID) AS num_concerts FROM stadium AS s JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Name | SELECT s.Name, COUNT(c.concert_ID) AS number_of_concerts FROM stadium AS s LEFT JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID, s.Name |

## Representative Examples
| variant | task_id | task_desc | react_success | variant_success | delta_reward | failure_mode | variant_memories |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | sql_66 | Show the stadium name and the number of concerts in each stadium. | 1 | 0 | -0.4400 | join inclusivity / zero-row inclusion | 84 (0.746): Find the total number of tours for each ranking date. \| issue=Query returned an empty result set, leading to confusion about data existence. \| learning=Recognize that an empty result set can be the correct answer if the underlying table is empty. \|\| 281 (0.722): Find the name of tourney that has more than 10 matches. \| issue=Not checking table emptiness before attempting complex queries. \| learning=Prioritize checking table row count with `SELECT COUNT(*)` to avoid misinterpreting empty results. |
| CR | sql_117 | Which airlines have less than 200 flights? | 1 | 1 | +0.0000 | neutral_query_change | 117 (0.594): What are the names and ids of every course with less than 2 sections? \| issue=Incorrectly counting sections using COUNT(*) instead of COUNT(section_id) after a JOIN. \| learning=When counting related entities after a JOIN, use COUNT(specific_id) to avoid counting duplicate rows from the join. |
| CR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | 1 | 0 | -0.7500 | set granularity / duplicate collapse | 170 (0.700): Return the country codes for countries that do not speak English. \| issue=Incorrect filtering logic: `WHERE Language != 'English'` returns countries that speak *... \| learning=To find countries that do not speak a specific language, use `NOT IN` with a subquery to exclude countries where that... |
| CR | sql_185 | What are the death and injury situations caused by the ship with tonnage 't'? | 1 | 1 | +0.0000 | unchanged | 77 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Incomplete SQL query leading to syntax error. \| learning=Construct a complete SQL SELECT statement with all necessary clauses (FROM, JOIN, ON). \|\| 78 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Missing JOIN conditions to link owners, dogs, and sizes tables. \| learning=Use JOIN clauses with appropriate ON conditions (owner_id, size_code) to link related tables. \|\| 79 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Not specifying columns to select from joined tables. \| learning=Explicitly select required columns (first_name, last_name, size_description) from the joined tables. |
| CR | sql_204 | which countries' tv channels are not playing any cartoon written by Todd Casey? | 0 | 1 | +0.6667 | improvement | 170 (0.529): Return the country codes for countries that do not speak English. \| issue=Incorrect filtering logic: `WHERE Language != 'English'` returns countries that speak *... \| learning=To find countries that do not speak a specific language, use `NOT IN` with a subquery to exclude countries where that... |
| TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | 1 | 0 | -0.4400 | prompt-only join inclusivity drift |  |
| TR | sql_117 | Which airlines have less than 200 flights? | 1 | 1 | +0.0000 | neutral_query_change |  |
| TR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | 1 | 1 | +0.0000 | neutral_query_change |  |
| TR | sql_185 | What are the death and injury situations caused by the ship with tonnage 't'? | 1 | 1 | +0.0000 | unchanged |  |
| TR | sql_204 | which countries' tv channels are not playing any cartoon written by Todd Casey? | 0 | 0 | +0.0000 | neutral_query_change |  |
| CR+TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | 1 | 0 | -0.4400 | join inclusivity / zero-row inclusion | 84 (0.746): Find the total number of tours for each ranking date. \| issue=Query returned an empty result set, leading to confusion about data existence. \| learning=Recognize that an empty result set can be the correct answer if the underlying table is empty. \|\| 281 (0.722): Find the name of tourney that has more than 10 matches. \| issue=Not checking table emptiness before attempting complex queries. \| learning=Prioritize checking table row count with `SELECT COUNT(*)` to avoid misinterpreting empty results. |
| CR+TR | sql_117 | Which airlines have less than 200 flights? | 1 | 0 | -1.0000 | join inclusivity / zero-row inclusion | 117 (0.594): What are the names and ids of every course with less than 2 sections? \| issue=Incorrectly counting sections using COUNT(*) instead of COUNT(section_id) after a JOIN. \| learning=When counting related entities after a JOIN, use COUNT(specific_id) to avoid counting duplicate rows from the join. |
| CR+TR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | 1 | 0 | -0.7500 | set granularity / duplicate collapse | 170 (0.700): Return the country codes for countries that do not speak English. \| issue=Incorrect filtering logic: `WHERE Language != 'English'` returns countries that speak *... \| learning=To find countries that do not speak a specific language, use `NOT IN` with a subquery to exclude countries where that... |
| CR+TR | sql_185 | What are the death and injury situations caused by the ship with tonnage 't'? | 1 | 0 | -1.0000 | output column drift | 77 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Incomplete SQL query leading to syntax error. \| learning=Construct a complete SQL SELECT statement with all necessary clauses (FROM, JOIN, ON). \|\| 78 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Missing JOIN conditions to link owners, dogs, and sizes tables. \| learning=Use JOIN clauses with appropriate ON conditions (owner_id, size_code) to link related tables. \|\| 79 (0.588): What are each owner's first name, last name, and the size of their dog? \| issue=Not specifying columns to select from joined tables. \| learning=Explicitly select required columns (first_name, last_name, size_description) from the joined tables. |
| CR+TR | sql_204 | which countries' tv channels are not playing any cartoon written by Todd Casey? | 0 | 0 | +0.0000 | neutral_query_change | 170 (0.529): Return the country codes for countries that do not speak English. \| issue=Incorrect filtering logic: `WHERE Language != 'English'` returns countries that speak *... \| learning=To find countries that do not speak a specific language, use `NOT IN` with a subquery to exclude countries where that... |
| hard-neg CR+TR | sql_66 | Show the stadium name and the number of concerts in each stadium. | 1 | 0 | -0.4400 | prompt-only join inclusivity drift | 272 (0.460): What is the official language used in the country the name of whose head of s... \| issue=Misinterpretation of 'the official language' when multiple official languages exist. \| learning=Re-evaluate query results and problem statement to identify the most appropriate answer. \|\| 85 (0.469): What is the average life expectancy in African countries that are republics? \| issue=Incorrectly using LIKE for exact string matching on GovernmentForm. \| learning=Use the exact equality operator (=) for precise string matching when the value is known. \|\| 88 (0.470): What languages are only used by a single country with a republic government? \| issue=Incorrect aggregation logic for counting distinct countries per language. \| learning=Use COUNT(T2.CountryCode) instead of COUNT(DISTINCT T1.Code) in HAVING clause for correct aggregation. |
| hard-neg CR+TR | sql_117 | Which airlines have less than 200 flights? | 1 | 1 | +0.0000 | neutral_query_change | 272 (0.419): What is the official language used in the country the name of whose head of s... \| issue=Misinterpretation of 'the official language' when multiple official languages exist. \| learning=Re-evaluate query results and problem statement to identify the most appropriate answer. |
| hard-neg CR+TR | sql_120 | find the pixel aspect ratio and nation of the tv channels that do not use English. | 1 | 1 | +0.0000 | neutral_query_change |  |
| hard-neg CR+TR | sql_185 | What are the death and injury situations caused by the ship with tonnage 't'? | 1 | 1 | +0.0000 | unchanged | 205 (0.450): Which language is the most popular on the Asian continent? \| issue=Incorrectly assuming `Percentage` column directly represents popularity across countries. \| learning=Aggregate `Percentage` by `Language` across all Asian countries to find overall popularity. \|\| 49 (0.450): How many countries have a republic as their form of government? \| issue=Incorrect filtering logic using LIKE instead of exact match for 'Republic'. \| learning=Use an exact match with '=' when the entire string value is known, instead of LIKE for specific values. |
| hard-neg CR+TR | sql_204 | which countries' tv channels are not playing any cartoon written by Todd Casey? | 0 | 0 | +0.0000 | neutral_query_change | 286 (0.416): Which owner owns the most dogs? List the owner id, first name and last name. \| issue=Incomplete SQL query leading to syntax error. \| learning=Construct a complete SQL query including SELECT, FROM, JOIN, GROUP BY, ORDER BY, and LIMIT clauses. |

## Suggested Figure/Table Additions
1. Add a paired SQL gate table with ReAct, CR, TR, CR+TR, and hard-neg CR+TR: success, average reward, regressions vs ReAct, improvements vs ReAct, and help calls.
2. Add a failure-mode taxonomy table with one representative query diff per mode: join inclusivity, duplicate policy, set granularity, output-column drift, and temporal normalization.
3. Add a compact paired-delta strip plot or heatmap over the 50 tasks, grouped by variant, to show that most tasks are unchanged and the underperformance is concentrated in a few brittle cases.
4. Add a near-miss vs hard-negative comparison panel: near-miss CR memories produce plausible wrong policies, while hard negatives mostly have lower uptake and fewer memory-induced failures.
5. Add a callout box for TR: zero help calls means the TR result is a prompt perturbation baseline, not evidence that retrieved tool instructions helped or hurt SQL execution.

## Outputs
- `experiment_reports/sql_transfer_radius/metrics_summary.csv`
- `experiment_reports/sql_transfer_radius/delta_status_summary.csv`
- `experiment_reports/sql_transfer_radius/failure_mode_summary.csv`
- `experiment_reports/sql_transfer_radius/delta_all_variants_vs_react.csv`
- `experiment_reports/sql_transfer_radius/regressions_vs_react.csv`
- `experiment_reports/sql_transfer_radius/representative_examples.csv`
