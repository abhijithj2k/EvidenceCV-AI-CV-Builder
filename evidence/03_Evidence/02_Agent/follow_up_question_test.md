# Follow-up Question Test

**Initial introduction:** My name is Alex Brown. My email is alex.brown@example.com. I use Python.

## Turn 1

**Action:** ask for `education`

**Question:** What qualification are you studying or have completed, and at which institution?

**User answer:** Bachelor of Computing Science at University of Technology Sydney

**Observation:** `education` stored only for `incomplete_user_01`; profile remains unconfirmed until review.

## Turn 2

**Action:** ask for `evidence`

**Question:** Describe one project or work experience that demonstrates your skills. What did you personally do?

**User answer:** Built a Python classifier and compared precision and recall on a labelled dataset.

**Observation:** `evidence` stored only for `incomplete_user_01`; profile remains unconfirmed until review.

## Turn 3

**Action:** ask for `job_goal`

**Question:** What specific role or job area are you targeting?

**User answer:** Graduate AI Engineer

**Observation:** `job_goal` stored only for `incomplete_user_01`; profile remains unconfirmed until review.

## Turn 4

**Action:** ask for `achievements`

**Question:** List any confirmed achievements, or answer ‘none’ if you have none to add.

**User answer:** none

**Observation:** `achievements` stored only for `incomplete_user_01`; profile remains unconfirmed until review.

**Outcome:** profile complete after 4 follow-up turns; test requires at least two. Pass.

A supplied complete profile produced zero follow-up questions. Pass.
