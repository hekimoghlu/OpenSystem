/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#include <sys/kern_memorystatus.h>

#include <darwintest.h>
#include <darwintest_utils.h>

#include "memorystatus_assertion_helpers.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"),
	T_META_CHECK_LEAKS(false),
	T_META_TAG_VM_PREFERRED
	);

#if ENTITLED
T_DECL(can_use_internal_bands_with_entitlement, "Can move process into internal bands with entitlement")
#else
T_DECL(can_not_use_internal_bands_without_entitlement, "Can not move process into internal bands with entitlement")
#endif
{
	for (int32_t band = JETSAM_PRIORITY_IDLE + 1; band <= JETSAM_PRIORITY_ENTITLED_MAX; band++) {
		int ret = set_priority(getpid(), band, 0, false);
		T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "set_priority");

		int32_t set_band, limit;
		uint64_t user_data;
		uint32_t state;
		bool success = get_priority_props(getpid(), false, &set_band, &limit, &user_data, &state);
		T_QUIET; T_ASSERT_TRUE(success, "get_priority_props");
#if ENTITLED
		T_QUIET; T_ASSERT_EQ(set_band, band, "Able to use entitled band");
#else
		T_QUIET; T_ASSERT_EQ(set_band, JETSAM_PRIORITY_IDLE, "Fell through to idle band");
#endif
	}
}
