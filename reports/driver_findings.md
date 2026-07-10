# Driver findings (plain English)


## enrollment_count (winner: Ridge)

- 5-fold R2: -0.130 +/- 0.160; LOOCV R2: -0.025; train-vs-CV R2 gap: 0.283.

**enrollment_count** — most influential features (permutation): CourseRating, inst_distinct_teachers, is_free, CourseCategory, CoursePrice.

- `CourseRating`: r=+0.29 with enrollment_count (higher CourseRating associates with higher enrollment_count).

- `inst_distinct_teachers`: r=-0.20 with enrollment_count (lower inst_distinct_teachers associates with lower enrollment_count).

- `is_free`: r=+0.19 with enrollment_count (higher is_free associates with higher enrollment_count).

- `CoursePrice`: r=-0.16 with enrollment_count (lower CoursePrice associates with lower enrollment_count).


## course_revenue (winner: Lasso)

- 5-fold R2: 0.991 +/- 0.004; LOOCV R2: 0.992; train-vs-CV R2 gap: 0.004.

**course_revenue** — most influential features (permutation): CoursePrice, CourseCategory, CourseRating, expertise_match_score, CourseLevel_ordinal.

- `CoursePrice`: r=+1.00 with course_revenue (higher CoursePrice associates with higher course_revenue).

- `CourseRating`: r=-0.02 with course_revenue (lower CourseRating associates with lower course_revenue).

- `expertise_match_score`: r=-0.11 with course_revenue (lower expertise_match_score associates with lower course_revenue).

- `CourseLevel_ordinal`: r=-0.07 with course_revenue (lower CourseLevel_ordinal associates with lower course_revenue).
