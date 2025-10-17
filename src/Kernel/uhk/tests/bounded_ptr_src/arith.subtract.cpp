/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
//  friend bounded_ptr operator-(bounded_ptr p, std::ptrdiff_t n);
//

#include <libkern/c++/bounded_ptr.h>
#include "test_utils.h"
#include <array>
#include <cstddef>
#include <darwintest.h>
#include <darwintest_utils.h>

#define _assert(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

struct T {
	int i;
};

template <typename T, typename QualT>
static void
tests()
{
	std::array<T, 5> array = {T{0}, T{1}, T{2}, T{3}, T{4}};

	// Subtract positive offsets
	// T{0}     T{1}     T{2}     T{3}     T{4}     <one-past-last>
	//   ^                                                ^
	//   |                                                |
	// begin                                           end,ptr
	{
		test_bounded_ptr<QualT> const ptr(array.end(), array.begin(), array.end());

		{
			test_bounded_ptr<QualT> res = ptr - static_cast<std::ptrdiff_t>(0);
			_assert(ptr == array.end());
		}
		{
			test_bounded_ptr<QualT> res = ptr - 1;
			_assert(&*res == &array[4]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - 2;
			_assert(&*res == &array[3]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - 3;
			_assert(&*res == &array[2]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - 4;
			_assert(&*res == &array[1]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - 5;
			_assert(&*res == &array[0]);
		}
	}

	// Subtract negative offsets
	// T{0}     T{1}     T{2}     T{3}     T{4}     <one-past-last>
	//   ^                                                ^
	//   |                                                |
	// begin,ptr                                         end
	{
		test_bounded_ptr<QualT> const ptr(array.begin(), array.begin(), array.end());

		{
			test_bounded_ptr<QualT> res = ptr - static_cast<std::ptrdiff_t>(0);
			_assert(&*res == &array[0]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - -1;
			_assert(&*res == &array[1]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - -2;
			_assert(&*res == &array[2]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - -3;
			_assert(&*res == &array[3]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - -4;
			_assert(&*res == &array[4]);
		}
		{
			test_bounded_ptr<QualT> res = ptr - -5;
			_assert(res == array.end());
		}
	}

	// Make sure the original pointer isn't modified
	{
		test_bounded_ptr<QualT> const ptr(array.begin() + 4, array.begin(), array.end());
		(void)(ptr - 2);
		_assert(&*ptr == &array[4]);
	}
}

T_DECL(arith_subtract, "bounded_ptr.arith.subtract", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
	tests<T, T volatile>();
	tests<T, T const volatile>();
}
