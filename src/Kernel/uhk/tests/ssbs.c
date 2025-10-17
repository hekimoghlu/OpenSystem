/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
/**
 * On devices that support it, this test ensures that PSTATE.SSBS is set by
 * default and is writeable by userspace.
 */
#include <darwintest.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <mach/thread_status.h>
#include <sys/sysctl.h>
#include <inttypes.h>


T_GLOBAL_META(
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("dmitry_grinberg"),
	T_META_RUN_CONCURRENTLY(true));


#define PSR64_SSBS              (0x1000)
#define REG_SSBS                "S3_3_C4_C2_6" /* clang will not emit MRS/MSR to "SSBS" itself since it doesnt always exist */

T_DECL(armv85_ssbs,
    "Test that ARMv8.5 SSBS is off by default (PSTATE.SSBS==1, don't ask!) and can be enabled by userspace.", T_META_TAG_VM_NOT_ELIGIBLE)
{
#ifndef __arm64__
	T_SKIP("Running on non-arm64 target, skipping...");
#else
	uint32_t ssbs_support = 19180;
	size_t ssbs_support_len = sizeof(ssbs_support);
	if (sysctlbyname("hw.optional.arm.FEAT_SSBS", &ssbs_support, &ssbs_support_len, NULL, 0)) {
		T_SKIP("Could not get SSBS support sysctl, skipping...");
	} else if (!ssbs_support) {
		T_SKIP("HW has no SSBS support, skipping...");
	} else if (ssbs_support != 1) {
		T_FAIL("SSBS support sysctl contains garbage: %u!", ssbs_support);
	} else {
		uint64_t ssbs_state = __builtin_arm_rsr64(REG_SSBS);

		if (!(ssbs_state & PSR64_SSBS)) {
			T_FAIL("SSBS does not default to off (value seen: 0x%" PRIx64 ")!", ssbs_state);
		}

		__builtin_arm_wsr64(REG_SSBS, 0);
		ssbs_state = __builtin_arm_rsr64(REG_SSBS);

		if (ssbs_state & PSR64_SSBS) {
			T_FAIL("SSBS did not turn on (value seen: 0x%" PRIx64 ")!", ssbs_state);
		}

		__builtin_arm_wsr64(REG_SSBS, PSR64_SSBS);
		ssbs_state = __builtin_arm_rsr64(REG_SSBS);

		if (!(ssbs_state & PSR64_SSBS)) {
			T_FAIL("SSBS did not turn off (value seen: 0x%" PRIx64 ")!", ssbs_state);
		}

		T_PASS("SSBS test passes");
	}
#endif /* __arm64__ */
}
