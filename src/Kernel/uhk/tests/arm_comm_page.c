/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#include <pthread.h>
#include <machine/cpu_capabilities.h>
#include <sys/types.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("jharmening"),
	T_META_RUN_CONCURRENTLY(true));

T_DECL(arm_comm_page_sanity,
    "Test that arm comm page values are sane.")
{
#if !defined(__arm64__)
	T_SKIP("Running on non-arm target, skipping...");
#else
	uint8_t page_shift = COMM_PAGE_READ(uint8_t, KERNEL_PAGE_SHIFT_LEGACY);
	T_QUIET; T_ASSERT_NE(page_shift, 0, "check that legacy kernel page shift is non-zero");
	T_QUIET; T_ASSERT_EQ(COMM_PAGE_READ(uint8_t, KERNEL_PAGE_SHIFT), page_shift,
	    "check that 'new' and 'legacy' page shifts are identical");
	T_QUIET; T_ASSERT_EQ(COMM_PAGE_READ(uint32_t, DEV_FIRM_LEGACY), COMM_PAGE_READ(uint32_t, DEV_FIRM),
	    "check that 'new' and 'legacy' DEV_FIRM fields are identical");
#endif
}
