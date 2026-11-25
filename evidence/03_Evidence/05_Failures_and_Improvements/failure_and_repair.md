# Failure and Repair

## Observed failure

The first `pytest` run produced **9 passes and 1 failure**. The incomplete-profile test raised `KeyError: 'name'` because `parse_introduction_offline` did not recognise the capitalised phrase “My name is Alex Brown.” The regular expression was case-sensitive.

## Repair

The explicit introductory phrase was made case-insensitive while the captured name retained capitalisation checks. No broader inference rule was added, so text such as “I am a third-year student” is not treated as a name.

## Expected improvement

The same introduction should capture `Alex Brown`; the next question should therefore be education rather than asking for the name again.

## Actual retest

The repeated test asked, in order: `education`, `evidence`, `job_goal`, and `achievements`. The repair run reported **10 passed**. Two later grounding and parser-regression checks increased the final suite to **12 passed**.

## Regression check

The five-profile retrieval test and cross-user isolation test were repeated in the same suite and remained successful.
