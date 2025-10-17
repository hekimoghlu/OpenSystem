/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include <sys/sysctl.h>
#include "apple_generic_timer.h"
#include "test_utils.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_ENABLED(TARGET_CPU_ARM64),
	T_META_OWNER("xi_han"),
	T_META_RUN_CONCURRENTLY(true),
	XNU_T_META_SOC_SPECIFIC
	);

#define AIDR_AGT (1ULL << 32)
T_DECL(apple_generic_timer,
    "Test that CNTFRQ_EL0 reads the correct frequency")
{
	uint64_t aidr;
	size_t sysctl_size = sizeof(aidr);
	sysctlbyname("machdep.cpu.sysreg_AIDR_EL1", &aidr, &sysctl_size, NULL, 0);

	const bool has_agt = aidr & AIDR_AGT;

	/* When AIDR_AGT is set, expect 1 GHz; otherwise expect 24 MHz. */
	agt_test_helper(has_agt);
}
