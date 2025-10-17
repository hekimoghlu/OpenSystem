/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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

#ifndef TESTS_BOUNDED_PTR_TEST_UTILS_H
#define TESTS_BOUNDED_PTR_TEST_UTILS_H

#include <cassert>
#include <libkern/c++/bounded_ptr.h>

namespace {
struct test_policy {
	static void
	trap(char const*)
	{
		assert(false);
	}
};

template <typename T>
using test_bounded_ptr = libkern::bounded_ptr<T, test_policy>;
} // end anonymous namespace

#endif // !TESTS_BOUNDED_PTR_TEST_UTILS_H
