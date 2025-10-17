/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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

//
// Tests for
//  explicit operator bool() const;
//

#include <libkern/c++/bounded_ptr.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include "test_utils.h"

#define _assert(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

struct T { };

template <typename T>
static void
tests()
{
	{
		test_bounded_ptr<T> p = nullptr;
		if (p) {
			_assert(false);
		}
		_assert(!p);
	}
	{
		T t;
		test_bounded_ptr<T> p(&t, &t, &t + 1);
		if (p) {
		} else {
			_assert(false);
		}
		_assert(!!p);
	}
}

T_DECL(operator_bool, "bounded_ptr.operator.bool", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
	tests<T volatile>();
	tests<T const volatile>();
}
