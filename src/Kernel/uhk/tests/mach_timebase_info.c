/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include <mach/mach_time.h>

#include <darwintest.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

extern kern_return_t mach_timebase_info_trap(mach_timebase_info_t info);

T_DECL(mach_timebase_info, "mach_timebase_info(_trap)",
    T_META_ALL_VALID_ARCHS(true), T_META_LTEPHASE(LTE_POSTINIT))
{
	mach_timebase_info_data_t a, b, c;

	T_ASSERT_EQ(KERN_SUCCESS, mach_timebase_info(&a), NULL);
	T_ASSERT_EQ(KERN_SUCCESS, mach_timebase_info(&b), NULL);
	T_ASSERT_EQ(KERN_SUCCESS, mach_timebase_info_trap(&c), NULL);

	T_EXPECT_EQ(a.numer, b.numer, NULL);
	T_EXPECT_EQ(a.denom, b.denom, NULL);
	T_EXPECT_EQ(a.numer, c.numer, NULL);
	T_EXPECT_EQ(a.denom, c.denom, NULL);
}
