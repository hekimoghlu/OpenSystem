/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#include <os/atomic_private.h>

T_GLOBAL_META(
	T_META_RUN_CONCURRENTLY(true),
	T_META_CHECK_LEAKS(false)
	);

T_DECL(os_atomic, "Just to make sure things build at all in c++ mode", T_META_TAG_VM_PREFERRED)
{
	static os_atomic(int) i;
	static volatile int v_i = 0;
	int old_i = 0;
	int a, b;

	T_ASSERT_EQ(os_atomic_inc_orig(&i, relaxed), 0, "atomic inc");
	T_ASSERT_EQ(os_atomic_cmpxchg(&i, 1, 0, relaxed), true, "os_atomic_cmpxchg");
	os_atomic_rmw_loop(&i, a, b, relaxed, {
		b = a;
	});

	T_ASSERT_EQ(os_atomic_inc_orig(&old_i, relaxed), 0, "atomic inc");
	T_ASSERT_EQ(os_atomic_cmpxchg(&old_i, 1, 0, relaxed), true, "os_atomic_cmpxchg");
	os_atomic_rmw_loop(&old_i, a, b, relaxed, {
		b = a;
	});

	T_ASSERT_EQ(os_atomic_inc_orig(&v_i, relaxed), 0, "atomic inc");
	T_ASSERT_EQ(os_atomic_cmpxchg(&v_i, 1, 0, relaxed), true, "os_atomic_cmpxchg");
	os_atomic_rmw_loop(&v_i, a, b, relaxed, {
		b = a;
	});
}
