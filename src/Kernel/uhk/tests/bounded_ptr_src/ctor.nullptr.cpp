/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
//  bounded_ptr(std::nullptr_t);
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
	// Test with nullptr
	{
		test_bounded_ptr<T> p = nullptr;
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p{nullptr};
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p(nullptr);
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p = static_cast<test_bounded_ptr<T> >(nullptr);
		_assert(p == nullptr);
	}
	{
		auto f = [](test_bounded_ptr<T> p) {
			    _assert(p == nullptr);
		    };
		f(nullptr);
	}

	// Test with NULL
	{
		test_bounded_ptr<T> p = NULL;
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p{NULL};
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p(NULL);
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p = static_cast<test_bounded_ptr<T> >(NULL);
		_assert(p == nullptr);
	}
	{
		auto f = [](test_bounded_ptr<T> p) {
			    _assert(p == nullptr);
		    };
		f(NULL);
	}

	// Test with 0
	{
		test_bounded_ptr<T> p = 0;
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p{0};
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p(0);
		_assert(p == nullptr);
	}
	{
		test_bounded_ptr<T> p = static_cast<test_bounded_ptr<T> >(0);
		_assert(p == nullptr);
	}
	{
		auto f = [](test_bounded_ptr<T> p) {
			    _assert(p == nullptr);
		    };
		f(0);
	}
}

T_DECL(ctor_nullptr, "bounded_ptr.ctor.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
	tests<T volatile>();
	tests<T const volatile>();
}
