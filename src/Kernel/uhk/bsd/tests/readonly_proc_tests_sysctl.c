/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#if DEVELOPMENT || DEBUG

#include <kern/startup.h>
#include <kern/zalloc.h>
#include <sys/proc_ro.h>
#include <sys/vm.h>

static int
readonly_proc_test_run(__unused int64_t in, int64_t *out)
{
	struct proc_ro *pro = proc_get_ro(current_proc());

	zone_require_ro(ZONE_ID_PROC_RO, sizeof(struct proc_ro), pro);

	*out = 1;
	return 0;
}

SYSCTL_TEST_REGISTER(readonly_proc_test, readonly_proc_test_run);

#endif /* DEVELOPMENT || DEBUG */
