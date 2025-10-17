/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
//  safe_allocation(std::nullptr_t);
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	{
		test_safe_allocation<T> array(nullptr);
		CHECK(array.data() == nullptr);
		CHECK(array.size() == 0);
		CHECK(array.begin() == array.end());
	}
	{
		test_safe_allocation<T> array{nullptr};
		CHECK(array.data() == nullptr);
		CHECK(array.size() == 0);
		CHECK(array.begin() == array.end());
	}
	{
		test_safe_allocation<T> array = nullptr;
		CHECK(array.data() == nullptr);
		CHECK(array.size() == 0);
		CHECK(array.begin() == array.end());
	}
	{
		auto f = [](test_safe_allocation<T> array) {
			    CHECK(array.data() == nullptr);
			    CHECK(array.size() == 0);
			    CHECK(array.begin() == array.end());
		    };
		f(nullptr);
	}
}

T_DECL(ctor_nullptr, "safe_allocation.ctor.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
