# Ground Truth Research Summary

## Status: Partial Completion

### Verified Commits (Ready for Evaluation)

| Task ID | Description | Commit After (Full SHA) | Commit Before (Full SHA) |
|---------|-------------|-------------------------|--------------------------|
| task_001 | Fix KeyError with importlib mode | `6486c3f3a858a0c8043f5c3f7c24297b82a0abe4` | `326faa25f4e776f082eea5603d84b0812b57773c` |
| task_002 | Fix pytest.approx Decimal crash | `7b21af10940143b74bb996c928b77bcb2d354c46` | `242522a10f59e4051c1f11a25c001fb2755eb2b1` |
| task_003 | Preserve source positions for AST | `40741c4aca50582cc9701ff01504b9e6dcd3396f` | `3d3ec5724c6f76bc07d0631ec8061f26f9ecac4c` |
| task_005 | Fix stream_with_context async | `9822a0351574790cb66c652fcc396ad7aa2b09d8` | `49b7e7bc8fb69d605719991d1c0a99fcee689053` |
| task_011 | Prevent annotation evaluation | `7021708b7c32a10dfb22ab30ff07d1dcd2b02f1b` | TODO: Find parent |
| task_019 | Exclude Pydantic fields with exclude_if | `7188388f99c089745e9ac30a0d60953024170cf8` | TODO: Find parent |
| task_020 | Add Jinja asyncio.run support | `5bc613ec45d51535849e7f8a67364a1fe1c94716` | TODO: Find parent |

### Tasks Needing Manual Research

| Task ID | Repository | Issue | Notes |
|---------|------------|-------|-------|
| task_004 | pallets/werkzeug | Commit references a PR branch (`emmanuelthome/fix-split-rn`) | Need to search for the merged commit |
| task_006 | pallets/flask | Commit references a PR branch (`test-redirect-session`) | Need to search for the merged commit |
| task_007 | pallets/jinja | Commit references a PR branch (`merged-1782`) | Need to search for the merged commit |
| task_008 | pallets/werkzeug | Short SHA + different PR branch | Found candidate `c7d2b8b2` but couldn't resolve full SHA |
| task_009 | pydantic/pydantic | Commit references a PR branch (`fix-12522`) | Need to search for the merged commit |
| task_010 | pydantic/pydantic | Commit references a PR branch (`viicos/fix-12421`) | Need to search for the merged commit |
| task_012 | pallets/click | Commit references a PR branch (`hanslenhert/fix-fish-quoted-params`) | Need to search for the merged commit |
| task_013 | pallets/click | Commit references a PR branch (`rolandwalker/fix-pager-args`) | Need to search for the merged commit |
| task_014 | fastapi/fastapi | Commit references a PR branch (`dataclass-merge`) | Need to search for the merged commit |
| task_015 | fastapi/fastapi | Commit references a PR branch (`recursion-merge`) | Need to search for the merged commit |
| task_016 | fastapi/fastapi | Commit references a PR branch (`hashable-merge`) | Need to search for the merged commit |
| task_017 | fastapi/fastapi | Short SHA `247ef32` | Need to find full SHA |
| task_018 | fastapi/fastapi | Short SHA `1c4fc96` | Need to find full SHA |

## Key Findings

1. **Many commits reference PR branches** that were merged into the main branch. The actual merged commit has a different SHA than the branch name.

2. **Some tasks already had valid 40-character SHAs** (task_011, task_019, task_020) - these are ready to use.

3. **The pytest repository** had the most success finding commits - task_001, task_002, and task_003 are all verified.

4. **The flask repository** had one verified commit (task_005).

## Recommended Next Steps

1. **Test the evaluation system** with the 4 verified tasks (task_001, task_002, task_003, task_005).

2. **For remaining tasks**, manually search each repository's GitHub page for:
   - PR numbers matching the task descriptions
   - The actual merged commit SHAs
   - Parent commits for the `commit_before` field

3. **Update the ground_truth.jsonl** file once all commits are verified.

## Files Created

- `fixtures/ground_truth_partial.jsonl` - Partial dataset with 4 verified tasks ready for testing
- `scripts/research_commits.py` - Automated research script (needs improvement for PR branch detection)
- `scripts/validate_and_fix_ground_truth.py` - Validation script for commit references

## Research Methodology Used

1. Cloned each repository
2. Searched git history using keywords from task descriptions
3. Checked commits that modified the expected files
4. Verified full 40-character SHAs
5. Found parent commits for `commit_before` values

## Limitations

The automated approach couldn't find commits that reference:
- PR branches from contributors (e.g., `emmanuelthome/fix-split-rn`)
- Merged PR references (e.g., `merged-1782`)
- Short SHAs that need expansion

These require manual GitHub/GitHub API lookup to find the actual merged commit SHAs.
