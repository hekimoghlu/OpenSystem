/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
//  template <typename T, typename U, typename R>
//  bool operator==(intrusive_shared_ptr<T, R> const& x, U* y);
//
//  template <typename T, typename U, typename R>
//  bool operator!=(intrusive_shared_ptr<T, R> const& x, U* y);
//
//  template <typename T, typename U, typename R>
//  bool operator==(T* x, intrusive_shared_ptr<U, R> const& y);
//
//  template <typename T, typename U, typename R>
//  bool operator!=(T* x, intrusive_shared_ptr<U, R> const& y);
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct Base { int i; };
struct Derived : Base { };

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
	T obj1{1};
	T obj2{2};

	{
		test_shared_ptr<TQual> const a(&obj1, libkern::no_retain);
		TQual* b = &obj2;
		check_ne(a, b);
	}

	{
		test_shared_ptr<TQual> const a(&obj1, libkern::no_retain);
		TQual* b = &obj1;
		check_eq(a, b);
	}

	{
		test_shared_ptr<TQual> const a = nullptr;
		TQual* b = &obj2;
		check_ne(a, b);
	}

	{
		test_shared_ptr<TQual> const a(&obj1, libkern::no_retain);
		TQual* b = nullptr;
		check_ne(a, b);
	}

	{
		test_shared_ptr<TQual> const a = nullptr;
		TQual* b = nullptr;
		check_eq(a, b);
	}
}

template <typename T, typename RelatedT>
static void
tests_convert()
{
	T obj{1};

	{
		test_shared_ptr<T> const a(&obj, libkern::no_retain);
		RelatedT* b = &obj;
		check_eq(a, b);
	}

	{
		test_shared_ptr<RelatedT> const a(&obj, libkern::no_retain);
		T* b = &obj;
		check_eq(a, b);
	}
}

T_DECL(compare_equal_raw, "intrusive_shared_ptr.compare.equal.raw", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
	tests_convert<Derived, Base>();
	tests_convert<Derived, Base const>();
}
