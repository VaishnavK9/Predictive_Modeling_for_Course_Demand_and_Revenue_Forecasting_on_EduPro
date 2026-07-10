# EduPro — Executive Summary
### Predicting course demand and revenue, and what actually drives them

**For:** EduPro leadership (non-technical) · **Scope:** 60 courses, 10,000
enrollments, calendar year 2025 · **Deliverables:** trained model suite,
interactive dashboard, full research paper.

---

## The one-sentence takeaway

**We can forecast course *revenue* very accurately — because it is driven almost
entirely by price — but course *demand* (how many people enroll) cannot be
predicted from the data we have today, because enrollments are nearly the same
for every course regardless of its attributes.**

---

## Top 5 insights

1. **Revenue is a pricing story, not a popularity story.** Our revenue model
   explains ~99% of the variation across courses, and price alone accounts for
   essentially all of it. Higher-priced paid courses make more money; everything
   else is a rounding error.

2. **Demand is flat across the catalog.** Every course attracts roughly the same
   number of enrollments (about 167, almost always between 140 and 196),
   independent of price, level, category, or instructor. No model could beat
   simply guessing the average — an honest and important result.

3. **Free vs. paid is the great divider.** Nearly two-thirds of enrollments are
   on free courses that earn **$0**. A course can be a popularity hit and a
   revenue zero, so demand and revenue must be managed as separate goals.

4. **Quality (rating) is the only lever that nudges demand** — and only weakly.
   Better-rated courses enroll slightly more; higher prices enroll slightly
   fewer. Useful as guidance, not as a precise forecast.

5. **There is little month-to-month seasonality.** Enrollment volume is broadly
   steady through the year; a monthly forecasting model only marginally beats
   "assume next month equals last month."

## Top 5 recommendations

1. **Use price as the primary revenue lever.** The dashboard's revenue forecast is
   reliable enough to test pricing scenarios before launch. Model revenue as
   *price × expected paid enrollments*.

2. **Stop forecasting demand course-by-course from current data.** Instead, plan
   with a demand *band* around the platform average, and avoid over-investing in
   precision that the data cannot support.

3. **Capture the features that actually drive demand.** Today's dataset lacks
   them. Start logging marketing spend, traffic source, search/recommendation
   placement, and prior-course behavior — these, not course attributes, are
   likely what move enrollments.

4. **Treat free courses as a funnel, not a loss.** Measure their worth by how many
   learners later buy paid courses (a metric we cannot yet compute), rather than
   by their direct revenue of zero.

5. **Lead revenue growth with higher-priced categories** (e.g., Artificial
   Intelligence) while using course quality/ratings to protect enrollment volume.

## Expected impact

- **Better pricing decisions, immediately.** A trustworthy revenue model lets the
  team compare launch and pricing options on expected revenue instead of
  intuition — the dashboard makes this a self-serve, minutes-long exercise.
- **Avoided waste.** Recognizing that demand is currently unpredictable prevents
  costly bets built on false-precision forecasts, and redirects effort toward
  collecting the data that would make demand predictable.
- **A foundation to build on.** The pipeline is reproducible and re-trainable; as
  richer production data arrives, the same framework will sharpen both forecasts.

## Honest caveats

- Conclusions rest on only **60 courses** for one year — directionally sound, but
  to be re-validated on production data.
- Revenue's predictability is partly **mechanical** (paid learners always pay the
  list price), not a behavioral discovery.
- The flat-demand pattern likely reflects how this (apparently synthetic) dataset
  was generated; real-world demand will vary more and should be re-modeled.

*Technical detail, full metrics, and figures: see `reports/research_paper.md` and
`reports/figures/`. Try scenarios yourself in the Streamlit dashboard.*
