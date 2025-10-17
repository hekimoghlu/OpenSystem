/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/kauth.h>
#include <security/mac_framework.h>
#include <security/mac_internal.h>

int
mac_necp_check_open(proc_t proc, int flags)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_proc_enforce) {
		return 0;
	}
#endif

	if (!mac_proc_check_enforce(proc)) {
		return 0;
	}

	MAC_CHECK(necp_check_open, current_cached_proc_cred(proc), flags);
	return error;
}

int
mac_necp_check_client_action(proc_t proc, struct fileglob *fg, uint32_t action)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_proc_enforce) {
		return 0;
	}
#endif

	if (!mac_proc_check_enforce(proc)) {
		return 0;
	}

	MAC_CHECK(necp_check_client_action, current_cached_proc_cred(proc), fg, action);
	return error;
}
