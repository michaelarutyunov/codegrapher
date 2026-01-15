# CodeGrapher Phase 12 Evaluation Results (Appendable)

This file accumulates results from individual evaluation runs. Each task is added independently as it's evaluated.

**Targets:** Token Savings ≥30%, Recall ≥85%, Precision ≤40%

---

## task_028: non-deterministic output from compile templates when using tuple unpacking

**Repo:** [jinja](https://github.com/pallets/jinja)
**Status:** ❌ FAIL
**Run Date:** 2026-01-14 14:56:02

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 66.9% | ≥30% | ✅ |
| Recall | 0.0% | ≥85% | ❌ |
| Precision | 0.0% | ≤40% | ✅ |

**Baseline Tokens:** 181,481
**CodeGrapher Tokens:** 60,079
**Files Returned:** 8

### Files

**Expected Files:** `src/jinja2/compiler.py`, `tests/test_compile.py`
- ✅ Found: None
- ❌ Missed: `tests/test_compile.py`, `src/jinja2/compiler.py`

**All Returned Files:** `src/jinja2/environment.py`, `tests/test_ext.py`, `src/jinja2/ext.py`, `src/jinja2/filters.py`, `src/jinja2/runtime.py`, `src/jinja2/nodes.py`, `src/jinja2/optimizer.py`, `src/jinja2/sandbox.py`

---

## task_032: Move utility functions from _utils.py to _client.py

**Repo:** [httpx](https://github.com/encode/httpx)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:02

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 81.2% | ≥30% | ✅ |
| Recall | 75.0% | ≥85% | ❌ |
| Precision | 42.9% | ≤40% | ❌ |

**Baseline Tokens:** 132,664
**CodeGrapher Tokens:** 24,882
**Files Returned:** 7

### Files

**Expected Files:** `httpx/_client.py`, `httpx/_utils.py`, `tests/client/test_headers.py`, `tests/test_utils.py`
- ✅ Found: `httpx/_client.py`, `httpx/_utils.py`, `tests/test_utils.py`
- ❌ Missed: `tests/client/test_headers.py`

**All Returned Files:** `tests/client/test_client.py`, `httpx/_client.py`, `tests/test_utils.py`, `tests/test_config.py`, `httpx/__init__.py`, `tests/client/test_async_client.py`, `httpx/_utils.py`

---

## task_039: Mark http_exception as async to prevent thread creation.

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:02

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 86.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 8.3% | ≤40% | ✅ |

**Baseline Tokens:** 137,770
**CodeGrapher Tokens:** 19,136
**Files Returned:** 12

### Files

**Expected Files:** `starlette/middleware/exceptions.py`, `tests/test_exceptions.py`
- ✅ Found: `starlette/middleware/exceptions.py`
- ❌ Missed: `tests/test_exceptions.py`

**All Returned Files:** `starlette/types.py`, `starlette/middleware/exceptions.py`, `starlette/applications.py`, `starlette/exceptions.py`, `starlette/_exception_handler.py`, `tests/test_concurrency.py`, `starlette/routing.py`, `tests/test_convertors.py`, `tests/test_config.py`, `starlette/middleware/base.py`, `starlette/requests.py`, `starlette/schemas.py`

---

## task_040: TestClient.request does not honor timeout

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:02

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 78.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 16.7% | ≤40% | ✅ |

**Baseline Tokens:** 135,481
**CodeGrapher Tokens:** 29,673
**Files Returned:** 6

### Files

**Expected Files:** `starlette/testclient.py`, `tests/test_testclient.py`
- ✅ Found: `starlette/testclient.py`
- ❌ Missed: `tests/test_testclient.py`

**All Returned Files:** `tests/test_applications.py`, `tests/test_endpoints.py`, `tests/test_exceptions.py`, `tests/test_routing.py`, `starlette/testclient.py`, `starlette/routing.py`

---

## task_028: non-deterministic output from compile templates when using tuple unpacking

**Repo:** [jinja](https://github.com/pallets/jinja)
**Status:** ❌ FAIL
**Run Date:** 2026-01-14 14:56:09

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 66.9% | ≥30% | ✅ |
| Recall | 0.0% | ≥85% | ❌ |
| Precision | 0.0% | ≤40% | ✅ |

**Baseline Tokens:** 181,481
**CodeGrapher Tokens:** 60,079
**Files Returned:** 8

### Files

**Expected Files:** `src/jinja2/compiler.py`, `tests/test_compile.py`
- ✅ Found: None
- ❌ Missed: `tests/test_compile.py`, `src/jinja2/compiler.py`

**All Returned Files:** `src/jinja2/environment.py`, `tests/test_ext.py`, `src/jinja2/ext.py`, `src/jinja2/filters.py`, `src/jinja2/runtime.py`, `src/jinja2/nodes.py`, `src/jinja2/optimizer.py`, `src/jinja2/sandbox.py`

---

## task_032: Move utility functions from _utils.py to _client.py

**Repo:** [httpx](https://github.com/encode/httpx)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:09

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 81.2% | ≥30% | ✅ |
| Recall | 75.0% | ≥85% | ❌ |
| Precision | 42.9% | ≤40% | ❌ |

**Baseline Tokens:** 132,664
**CodeGrapher Tokens:** 24,882
**Files Returned:** 7

### Files

**Expected Files:** `httpx/_client.py`, `httpx/_utils.py`, `tests/client/test_headers.py`, `tests/test_utils.py`
- ✅ Found: `httpx/_client.py`, `tests/test_utils.py`, `httpx/_utils.py`
- ❌ Missed: `tests/client/test_headers.py`

**All Returned Files:** `tests/client/test_client.py`, `httpx/_client.py`, `tests/test_utils.py`, `tests/test_config.py`, `httpx/__init__.py`, `tests/client/test_async_client.py`, `httpx/_utils.py`

---

## task_039: Mark http_exception as async to prevent thread creation.

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:09

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 86.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 8.3% | ≤40% | ✅ |

**Baseline Tokens:** 137,770
**CodeGrapher Tokens:** 19,136
**Files Returned:** 12

### Files

**Expected Files:** `starlette/middleware/exceptions.py`, `tests/test_exceptions.py`
- ✅ Found: `starlette/middleware/exceptions.py`
- ❌ Missed: `tests/test_exceptions.py`

**All Returned Files:** `starlette/types.py`, `starlette/middleware/exceptions.py`, `starlette/applications.py`, `starlette/exceptions.py`, `starlette/_exception_handler.py`, `tests/test_concurrency.py`, `starlette/routing.py`, `tests/test_convertors.py`, `tests/test_config.py`, `starlette/middleware/base.py`, `starlette/requests.py`, `starlette/schemas.py`

---

## task_040: TestClient.request does not honor timeout

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:09

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 78.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 16.7% | ≤40% | ✅ |

**Baseline Tokens:** 135,481
**CodeGrapher Tokens:** 29,673
**Files Returned:** 6

### Files

**Expected Files:** `starlette/testclient.py`, `tests/test_testclient.py`
- ✅ Found: `starlette/testclient.py`
- ❌ Missed: `tests/test_testclient.py`

**All Returned Files:** `tests/test_applications.py`, `tests/test_endpoints.py`, `tests/test_exceptions.py`, `tests/test_routing.py`, `starlette/testclient.py`, `starlette/routing.py`

---

## task_028: non-deterministic output from compile templates when using tuple unpacking

**Repo:** [jinja](https://github.com/pallets/jinja)
**Status:** ❌ FAIL
**Run Date:** 2026-01-14 14:56:15

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 66.9% | ≥30% | ✅ |
| Recall | 0.0% | ≥85% | ❌ |
| Precision | 0.0% | ≤40% | ✅ |

**Baseline Tokens:** 181,481
**CodeGrapher Tokens:** 60,079
**Files Returned:** 8

### Files

**Expected Files:** `src/jinja2/compiler.py`, `tests/test_compile.py`
- ✅ Found: None
- ❌ Missed: `tests/test_compile.py`, `src/jinja2/compiler.py`

**All Returned Files:** `src/jinja2/environment.py`, `tests/test_ext.py`, `src/jinja2/ext.py`, `src/jinja2/filters.py`, `src/jinja2/runtime.py`, `src/jinja2/nodes.py`, `src/jinja2/optimizer.py`, `src/jinja2/sandbox.py`

---

## task_032: Move utility functions from _utils.py to _client.py

**Repo:** [httpx](https://github.com/encode/httpx)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:15

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 81.2% | ≥30% | ✅ |
| Recall | 75.0% | ≥85% | ❌ |
| Precision | 42.9% | ≤40% | ❌ |

**Baseline Tokens:** 132,664
**CodeGrapher Tokens:** 24,882
**Files Returned:** 7

### Files

**Expected Files:** `httpx/_client.py`, `httpx/_utils.py`, `tests/client/test_headers.py`, `tests/test_utils.py`
- ✅ Found: `httpx/_client.py`, `httpx/_utils.py`, `tests/test_utils.py`
- ❌ Missed: `tests/client/test_headers.py`

**All Returned Files:** `tests/client/test_client.py`, `httpx/_client.py`, `tests/test_utils.py`, `tests/test_config.py`, `httpx/__init__.py`, `tests/client/test_async_client.py`, `httpx/_utils.py`

---

## task_039: Mark http_exception as async to prevent thread creation.

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:15

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 86.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 8.3% | ≤40% | ✅ |

**Baseline Tokens:** 137,770
**CodeGrapher Tokens:** 19,136
**Files Returned:** 12

### Files

**Expected Files:** `starlette/middleware/exceptions.py`, `tests/test_exceptions.py`
- ✅ Found: `starlette/middleware/exceptions.py`
- ❌ Missed: `tests/test_exceptions.py`

**All Returned Files:** `starlette/types.py`, `starlette/middleware/exceptions.py`, `starlette/applications.py`, `starlette/exceptions.py`, `starlette/_exception_handler.py`, `tests/test_concurrency.py`, `starlette/routing.py`, `tests/test_convertors.py`, `tests/test_config.py`, `starlette/middleware/base.py`, `starlette/requests.py`, `starlette/schemas.py`

---

## task_040: TestClient.request does not honor timeout

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:15

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 78.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 16.7% | ≤40% | ✅ |

**Baseline Tokens:** 135,481
**CodeGrapher Tokens:** 29,673
**Files Returned:** 6

### Files

**Expected Files:** `starlette/testclient.py`, `tests/test_testclient.py`
- ✅ Found: `starlette/testclient.py`
- ❌ Missed: `tests/test_testclient.py`

**All Returned Files:** `tests/test_applications.py`, `tests/test_endpoints.py`, `tests/test_exceptions.py`, `tests/test_routing.py`, `starlette/testclient.py`, `starlette/routing.py`

---

## task_028: non-deterministic output from compile templates when using tuple unpacking

**Repo:** [jinja](https://github.com/pallets/jinja)
**Status:** ❌ FAIL
**Run Date:** 2026-01-14 14:56:21

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 66.9% | ≥30% | ✅ |
| Recall | 0.0% | ≥85% | ❌ |
| Precision | 0.0% | ≤40% | ✅ |

**Baseline Tokens:** 181,481
**CodeGrapher Tokens:** 60,079
**Files Returned:** 8

### Files

**Expected Files:** `src/jinja2/compiler.py`, `tests/test_compile.py`
- ✅ Found: None
- ❌ Missed: `tests/test_compile.py`, `src/jinja2/compiler.py`

**All Returned Files:** `src/jinja2/environment.py`, `tests/test_ext.py`, `src/jinja2/ext.py`, `src/jinja2/filters.py`, `src/jinja2/runtime.py`, `src/jinja2/nodes.py`, `src/jinja2/optimizer.py`, `src/jinja2/sandbox.py`

---

## task_032: Move utility functions from _utils.py to _client.py

**Repo:** [httpx](https://github.com/encode/httpx)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:21

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 81.2% | ≥30% | ✅ |
| Recall | 75.0% | ≥85% | ❌ |
| Precision | 42.9% | ≤40% | ❌ |

**Baseline Tokens:** 132,664
**CodeGrapher Tokens:** 24,882
**Files Returned:** 7

### Files

**Expected Files:** `httpx/_client.py`, `httpx/_utils.py`, `tests/client/test_headers.py`, `tests/test_utils.py`
- ✅ Found: `tests/test_utils.py`, `httpx/_utils.py`, `httpx/_client.py`
- ❌ Missed: `tests/client/test_headers.py`

**All Returned Files:** `tests/client/test_client.py`, `httpx/_client.py`, `tests/test_utils.py`, `tests/test_config.py`, `httpx/__init__.py`, `tests/client/test_async_client.py`, `httpx/_utils.py`

---

## task_039: Mark http_exception as async to prevent thread creation.

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:21

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 86.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 8.3% | ≤40% | ✅ |

**Baseline Tokens:** 137,770
**CodeGrapher Tokens:** 19,136
**Files Returned:** 12

### Files

**Expected Files:** `starlette/middleware/exceptions.py`, `tests/test_exceptions.py`
- ✅ Found: `starlette/middleware/exceptions.py`
- ❌ Missed: `tests/test_exceptions.py`

**All Returned Files:** `starlette/types.py`, `starlette/middleware/exceptions.py`, `starlette/applications.py`, `starlette/exceptions.py`, `starlette/_exception_handler.py`, `tests/test_concurrency.py`, `starlette/routing.py`, `tests/test_convertors.py`, `tests/test_config.py`, `starlette/middleware/base.py`, `starlette/requests.py`, `starlette/schemas.py`

---

## task_040: TestClient.request does not honor timeout

**Repo:** [starlette](https://github.com/encode/starlette)
**Status:** ⚠️  PARTIAL
**Run Date:** 2026-01-14 14:56:21

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 78.1% | ≥30% | ✅ |
| Recall | 50.0% | ≥85% | ❌ |
| Precision | 16.7% | ≤40% | ✅ |

**Baseline Tokens:** 135,481
**CodeGrapher Tokens:** 29,673
**Files Returned:** 6

### Files

**Expected Files:** `starlette/testclient.py`, `tests/test_testclient.py`
- ✅ Found: `starlette/testclient.py`
- ❌ Missed: `tests/test_testclient.py`

**All Returned Files:** `tests/test_applications.py`, `tests/test_endpoints.py`, `tests/test_exceptions.py`, `tests/test_routing.py`, `starlette/testclient.py`, `starlette/routing.py`

---

