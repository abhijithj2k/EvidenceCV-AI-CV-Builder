# Cross-user Separation Test

`ProfileStore` saved `student_profile_02` and `student_profile_05` under separate dictionary keys and returned defensive copies.

| Check | Result |
|---|---|
| User 1 name appears in User 2 JSON | Pass — absent |
| User 2 name appears in User 1 JSON | Pass — absent |
| CV 1 user_id | `student_profile_02` |
| CV 3 user_id | `student_profile_05` |
| Agent cross-user request | Blocked with `cross-user access blocked` |

Changing a target job creates a new `CVDocument`; it does not write back to `ConfirmedProfile`.
