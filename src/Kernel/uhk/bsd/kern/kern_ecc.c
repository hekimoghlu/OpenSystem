/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#include <kern/debug.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/priv.h>
#include <mach/machine.h>
#include <libkern/libkern.h>
#include <kern/assert.h>
#include <pexpert/pexpert.h>
#include <kern/ecc.h>

static int
get_ecc_data_handler(__unused struct sysctl_oid *oidp, __unused void *arg1, __unused int arg2,
    struct sysctl_req *req)
{
	struct ecc_event ev;
	int changed, retval;

	if (priv_check_cred(kauth_cred_get(), PRIV_HW_DEBUG_DATA, 0) != 0) {
		return EPERM;
	}

	if (KERN_SUCCESS != ecc_log_get_next_event(&ev)) {
		/*
		 * EAGAIN would be better, but sysctl infrastructure
		 * interprets that */
		return EBUSY;
	}

	retval = sysctl_io_opaque(req, &ev, sizeof(ev), &changed);
	assert(!changed);

	return retval;
}

SYSCTL_PROC(_kern, OID_AUTO, next_ecc_event,
    CTLFLAG_RD | CTLFLAG_ANYBODY | CTLFLAG_MASKED | CTLTYPE_STRUCT,
    0, 0, get_ecc_data_handler,
    "-", "");
