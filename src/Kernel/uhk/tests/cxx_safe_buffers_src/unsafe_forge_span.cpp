/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
//  __unsafe_forge_span functions
//

#include <span>
#include <vector>
#include <os/cxx_safe_buffers.h>
#include <darwintest.h>

#define CHECK(...) T_ASSERT_TRUE((__VA_ARGS__), # __VA_ARGS__)

struct S {
	int i;
};

template <typename T>
static void
tests()
{
	{
		T * p = new T[10];
		std::span<T> span = os::span::__unsafe_forge_span(p, 10);

		CHECK(span.data() == p && span.size() == 10);
		delete[] p;
	}
	{
		const T * p = new T[10];
		std::span<const T> span = os::span::__unsafe_forge_span(p, 10);

		CHECK(span.data() == p && span.size() == 10);
		delete[] p;
	}
	{
		std::vector<T> v;
		std::span<T> span = os::span::__unsafe_forge_span(v.begin(), v.end());

		CHECK(span.data() == v.data() && span.size() == 0);
	}
	{
		T * p = new T[10];
		std::span<T> span = os::unsafe_forge_span(p, 10);
		std::span<T, 10> span2 = os::unsafe_forge_span<T, 10>(p);

		CHECK(span.data() == p && span.size() == 10);
		CHECK(span2.data() == p && span2.size() == 10);
		delete[] p;
	}
	{
		std::vector<T> v;
		std::span<T> span = os::unsafe_forge_span(v.begin(), v.end());

		CHECK(span.data() == v.data() && span.size() == 0);
	}
}

T_DECL(unsafe_forge_span, "cxx_safe_buffers.unsafe_forge_span")
{
	tests<S>();
}
