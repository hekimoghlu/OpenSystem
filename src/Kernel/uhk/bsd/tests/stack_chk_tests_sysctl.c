/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#if (DEVELOPMENT || DEBUG) && !KASAN

#include <sys/sysctl.h>
#include <libkern/stack_protector.h>

__attribute__((noinline))
static int
check_for_cookie(size_t len)
{
	long buf[4];
	long *search = (long *)(void *)&buf[0];
	size_t n;

	assert(len < sizeof(buf));
	assert(__stack_chk_guard != 0);
	assert(((uintptr_t)search & (sizeof(long) - 1)) == 0);

	/* force compiler to insert stack cookie check: */
	memset_s(buf, len, 0, len);

	/* 32 x sizeof(long) should be plenty to find the cookie: */
	for (n = 0; n < 32; ++n) {
		if (*(search++) == __stack_chk_guard) {
			return 0;
		}
	}

	return ESRCH;
}

static int
sysctl_run_stack_chk_tests SYSCTL_HANDLER_ARGS
{
	#pragma unused(arg1, arg2, oidp)

	unsigned int dummy = 0;
	int error, changed = 0, kr;
	error = sysctl_io_number(req, 0, sizeof(dummy), &dummy, &changed);
	if (error || !changed) {
		return error;
	}

	kr = check_for_cookie(3);
	return kr;
}

SYSCTL_PROC(_kern, OID_AUTO, run_stack_chk_tests,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED | CTLFLAG_MASKED,
    0, 0, sysctl_run_stack_chk_tests, "I", "");

#endif /* (DEVELOPMENT || DEBUG) && !KASAN */
