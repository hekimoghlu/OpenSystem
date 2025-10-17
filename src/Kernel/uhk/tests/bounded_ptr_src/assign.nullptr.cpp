/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
//  bounded_ptr& operator=(std::nullptr_t);
//

#include <cstddef>
#include <libkern/c++/bounded_ptr.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include "test_utils.h"

#define _assert(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

struct T { };

template <typename T, typename TQual>
static void
tests()
{
	T obj{};

	// Assign from nullptr
	{
		test_bounded_ptr<TQual> p(&obj, &obj, &obj + 1);
		_assert(p != nullptr);
		test_bounded_ptr<TQual>& ref = (p = nullptr);
		_assert(&ref == &p);
		_assert(p == nullptr);
	}

	// Assign from NULL
	{
		test_bounded_ptr<TQual> p(&obj, &obj, &obj + 1);
		_assert(p != nullptr);
		test_bounded_ptr<TQual>& ref = (p = NULL);
		_assert(&ref == &p);
		_assert(p == nullptr);
	}

	// Assign from 0
	{
		test_bounded_ptr<TQual> p(&obj, &obj, &obj + 1);
		_assert(p != nullptr);
		test_bounded_ptr<TQual>& ref = (p = 0);
		_assert(&ref == &p);
		_assert(p == nullptr);
	}
}

T_DECL(assign_nullptr, "bounded_ptr.assign.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
	tests<T, T volatile>();
	tests<T, T const volatile>();
}
