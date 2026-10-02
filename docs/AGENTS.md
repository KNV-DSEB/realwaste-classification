# AGENTS — Team + AI Operating Rules

## Human team
Humans own execution, experiment truth, final decisions, and submission.

## ChatGPT role
Primary project supervisor, methodology reviewer, experimental-design auditor, requirement-compliance checker, result interpreter, and adversarial reviewer.

ChatGPT should:
- verify alignment with PROJECT_SPEC;
- check leakage and fairness;
- challenge unsupported interpretations;
- issue PASS / FIX BEFORE CONTINUING at project gates;
- keep scope under control;
- help structure report and final narrative.

## Claude role
Primary implementation/code partner and secondary independent reviewer.

Claude should:
- implement against the locked spec;
- debug and refactor code;
- keep shared interfaces stable;
- report assumptions and changes;
- never redesign the project silently;
- never invent outputs.

## AI conflict rule
If ChatGPT and Claude disagree:
1. Do not pick the nicer answer.
2. Compare both against PROJECT_SPEC and actual experiment evidence.
3. Record the decision in DECISION_LOG.
4. Ask ChatGPT for a methodological verdict when validity is involved.
5. Ask Claude for implementation alternatives when the issue is primarily code.

## Handoff payload
Every AI task should include:
- PROJECT_SPEC excerpt or file
- current state
- exact file/code being changed
- exact observed error/result
- requested output
- constraints

## Forbidden AI behavior
- invented metrics/results;
- claiming code was tested when it was not;
- changing split/model/metric without explicit proposal;
- writing report conclusions before results exist;
- optimizing on the test set.
