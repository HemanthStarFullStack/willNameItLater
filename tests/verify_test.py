"""HHEM sanity: entailed answers score high, contradictions score low —
including negation, which lexical overlap is structurally blind to.

Run inside the container:
    docker exec ondevice-ai python tests/verify_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify
from config import HHEM_THRESHOLD

CASES = [
    # (evidence, claim, should_be_grounded)
    ("[1] (Health) My blood group is O positive.",
     "Your blood group is O positive.", True),
    ("[1] (Health) My blood group is O positive.",
     "Your blood group is B negative.", False),
    # 100% lexical overlap, opposite meaning — the reason HHEM exists.
    ("[1] (Health) I'm allergic to penicillin.",
     "You are not allergic to penicillin.", False),
    ("[1] (Work) My manager is Priya; standup is 10am daily on weekdays.",
     "Your standup is at 10am.", True),
    # Wrong-fact substitution (the old passport bug).
    ("[1] (Personal) Passport number X1234567, expires in 2029.",
     "Your name is Hemanth.", False),
]

fails = 0
for evidence, claim, want in CASES:
    s = verify.score(evidence, claim)
    got = s >= HHEM_THRESHOLD
    ok = got == want
    fails += not ok
    print(f'{"PASS" if ok else "FAIL"}  HHEM={s:.3f}  '
          f'(want {"grounded" if want else "hallucination"})  {claim!r}')

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
