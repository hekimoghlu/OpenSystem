/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include <sys/sysctl.h>

#if defined(KERNEL_INTEGRITY_CTRR) && defined(CONFIG_XNUPOST)
extern kern_return_t ctrr_test(void);

static int
sysctl_run_ctrr_test(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2, struct sysctl_req *req)
{
	unsigned int dummy;
	int error, changed;
	error = sysctl_io_number(req, 0, sizeof(dummy), &dummy, &changed);
	if (error || !changed) {
		return error;
	}
	return ctrr_test();
}

SYSCTL_PROC(_kern, OID_AUTO, run_ctrr_test,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED,
    0, 0, sysctl_run_ctrr_test, "I", "");
#endif /* defined(KERNEL_INTEGRITY_CTRR) && defined(CONFIG_XNUPOST) */
