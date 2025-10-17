/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include <sys/types.h>
#include <sys/sysctl.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.arm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("arm"),
	T_META_OWNER("joster"),
	T_META_RUN_CONCURRENTLY(true),
	T_META_TAG_VM_PREFERRED);

T_DECL(x18_unentitled,
    "Test that x18 is trashed when unentitled.")
{
#ifndef __arm64__
	T_SKIP("Running on non-arm64 target, skipping...");
#else
	uint64_t x18_val;
	for (uint64_t i = 0xFEEDB0B000000000ULL; i < 0xFEEDB0B000000000ULL + 10000; ++i) {
		asm volatile ("mov x18, %0" : : "r"(i));
		sched_yield();
		asm volatile ("mov %0, x18" : "=r"(x18_val));
		T_QUIET; T_ASSERT_EQ(x18_val, 0ULL, "check that x18 is cleared after yield");
	}
#endif
}
