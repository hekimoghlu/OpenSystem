/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
#include <uuid/uuid.h>
#include <System/sys/proc_uuid_policy.h>
#include <stdint.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

#define NUM_PROC_UUID_POLICY_FLAGS 4

T_DECL(proc_uuid_policy_26567533, "Tests passing a NULL uuid in (uap->uuid).", T_META_LTEPHASE(LTE_POSTINIT), T_META_TAG_VM_PREFERRED)
{
	int i, ret;
	uuid_t null_uuid;
	memset(null_uuid, 0, sizeof(uuid_t));

	uint32_t policy_flags[] = {
		PROC_UUID_POLICY_FLAGS_NONE,
		PROC_UUID_NO_CELLULAR,
		PROC_UUID_NECP_APP_POLICY,
		PROC_UUID_ALT_DYLD_POLICY
	};

	for (i = 0; i < NUM_PROC_UUID_POLICY_FLAGS; i++) {
		T_LOG("Testing policy add with flag value 0x%x", policy_flags[i]);

		/* Since UUID is null, this call should fail with errno = EINVAL. */
		ret = proc_uuid_policy(PROC_UUID_POLICY_OPERATION_ADD, null_uuid, sizeof(uuid_t), policy_flags[i]);

		T_ASSERT_TRUE(ret == -1, "proc_uuid_policy returned %d", ret);
		T_WITH_ERRNO;
		T_ASSERT_TRUE(errno = EINVAL, "errno is %d", errno);
	}

	for (i = 0; i < NUM_PROC_UUID_POLICY_FLAGS; i++) {
		T_LOG("Testing policy remove with flag value 0x%x", policy_flags[i]);

		/* Since UUID is null, this call should fail with errno = EINVAL. */
		ret = proc_uuid_policy(PROC_UUID_POLICY_OPERATION_REMOVE, null_uuid, sizeof(uuid_t), policy_flags[i]);

		T_ASSERT_TRUE(ret == -1, "proc_uuid_policy returned %d", ret);
		T_WITH_ERRNO;
		T_ASSERT_TRUE(errno = EINVAL, "errno is %d", errno);
	}
}
