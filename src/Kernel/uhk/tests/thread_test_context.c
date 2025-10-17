/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <darwintest.h>
#include <darwintest_utils.h>
#include <test_utils.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.misc"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_CHECK_LEAKS(false),
	T_META_OWNER("gparker"));

static int64_t
run_sysctl_test(const char *t, int64_t value)
{
	char name[1024];
	int64_t result = 0;
	size_t s = sizeof(value);
	int rc;

	snprintf(name, sizeof(name), "debug.test.%s", t);
	rc = sysctlbyname(name, &result, &s, &value, s);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(rc, "sysctlbyname(%s)", t);
	return result;
}

T_DECL(thread_test_context,
    "infrastructure for threads running kernel tests",
    XNU_T_META_REQUIRES_DEVELOPMENT_KERNEL)
{
	int64_t bad_line = run_sysctl_test("thread_test_context", 0);
	/* return value is one or two line numbers in thread.c */
	int64_t bad_line_2 = bad_line >> 32;
	bad_line = (bad_line << 32) >> 32;

	if (bad_line_2) {
		T_FAIL("error at osfmk/kern/thread.c:%lld from thread.c:%lld",
		    bad_line_2, bad_line);
	} else if (bad_line) {
		T_FAIL("error at osfmk/kern/thread.c:%lld",
		    bad_line);
	} else {
		T_PASS("thread_test_context");
	}
}
