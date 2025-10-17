/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
//  template <typename T, typename R>
//  bool operator==(intrusive_shared_ptr<T, R> const& x, std::nullptr_t);
//
//  template <typename T, typename R>
//  bool operator!=(intrusive_shared_ptr<T, R> const& x, std::nullptr_t);
//
//  template <typename T, typename R>
//  bool operator==(std::nullptr_t, intrusive_shared_ptr<T, R> const& x);
//
//  template <typename T, typename R>
//  bool operator!=(std::nullptr_t, intrusive_shared_ptr<T, R> const& x);
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct T { int i; };

template <typename T, typename U>
static void
check_eq(T t, U u)
{
	CHECK(t == u);
	CHECK(u == t);
	CHECK(!(t != u));
	CHECK(!(u != t));
}

template <typename T, typename U>
static void
check_ne(T t, U u)
{
	CHECK(!(t == u));
	CHECK(!(u == t));
	CHECK(t != u);
	CHECK(u != t);
}

template <typename T, typename TQual>
static void
tests()
{
	T obj{3};

	{
		test_shared_ptr<TQual> const a(&obj, libkern::no_retain);
		check_ne(a, nullptr);
	}

	{
		test_shared_ptr<TQual> const a = nullptr;
		check_eq(a, nullptr);
	}
}

T_DECL(compare_equal_nullptr, "intrusive_shared_ptr.compare.equal.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
}
