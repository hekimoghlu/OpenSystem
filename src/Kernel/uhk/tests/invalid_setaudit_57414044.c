/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <bsm/audit.h>
#include <bsm/audit_session.h>
#include <err.h>
#include <sysexits.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(invalid_setaudit_57414044,
    "Verify that auditing a setaudit_addr syscall which has an invalid "
    "at_type field does not panic",
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	T_SETUPBEGIN;

	int cond, ret = auditon(A_GETCOND, &cond, sizeof(cond));
	if (ret == -1 && errno == ENOSYS) {
		T_SKIP("no kernel support for auditing; can't test");
	}
	T_ASSERT_POSIX_SUCCESS(ret, "auditon A_GETCOND");
	if (cond != AUC_AUDITING) {
		T_SKIP("auditing is not enabled; can't test");
	}

	/* set up auditing to audit `setaudit_addr` */
	auditpinfo_addr_t pinfo_addr = {.ap_pid = getpid()};
	T_ASSERT_POSIX_SUCCESS(auditon(A_GETPINFO_ADDR, &pinfo_addr, sizeof(pinfo_addr)), NULL);
	auditpinfo_t pinfo = {.ap_pid = getpid(), .ap_mask = pinfo_addr.ap_mask};
	pinfo.ap_mask.am_failure |= 0x800; /* man 5 audit_class */
	T_ASSERT_POSIX_SUCCESS(auditon(A_SETPMASK, &pinfo, sizeof(pinfo)), NULL);

	T_SETUPEND;

	struct auditinfo_addr a;
	memset(&a, 0, sizeof(a));
	a.ai_termid.at_type = 999;
	T_ASSERT_POSIX_FAILURE(setaudit_addr(&a, sizeof(a)), EINVAL,
	    "setaudit_addr should fail due to invalid at_type");
}
