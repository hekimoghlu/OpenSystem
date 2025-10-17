/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

#ifndef TESTS_INTRUSIVE_SHARED_PTR_TEST_POLICY_H
#define TESTS_INTRUSIVE_SHARED_PTR_TEST_POLICY_H

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest_utils.h>

struct test_policy {
	static inline int retain_count = 0;

	template <typename T>
	static void
	retain(T&)
	{
		++retain_count;
	}
	template <typename T>
	static void
	release(T&)
	{
		--retain_count;
	}
};

struct tracking_policy {
	static inline int retains = 0;
	static inline int releases = 0;
	static inline int refcount = 0;
	static inline bool hit_zero = false;

	static void
	reset()
	{
		retains = 0;
		releases = 0;
		refcount = 0;
		hit_zero = false;
	}

	template <typename T>
	static void
	retain(T&)
	{
		++retains;
		++refcount;
	}
	template <typename T>
	static void
	release(T&)
	{
		++releases;
		--refcount;
		if (refcount == 0) {
			hit_zero = true;
		}
	}
};

template <int>
struct dummy_policy {
	template <typename T>
	static void
	retain(T&)
	{
	}
	template <typename T>
	static void
	release(T&)
	{
	}
};

template <typename T>
using tracked_shared_ptr = libkern::intrusive_shared_ptr<T, tracking_policy>;

template <typename T>
using test_shared_ptr = libkern::intrusive_shared_ptr<T, test_policy>;

#define CHECK(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

#endif // !TESTS_INTRUSIVE_SHARED_PTR_TEST_POLICY_H
