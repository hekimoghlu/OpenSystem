/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
//  explicit safe_allocation(size_t n, allocate_memory_zero_t);
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
		test_safe_allocation<T> const array(10, libkern::allocate_memory_zero);
		CHECK(array.data() != nullptr);
		CHECK(array.size() == 10);
		CHECK(array.begin() == array.data());
		CHECK(array.end() == array.data() + 10);

		auto const byteArray = reinterpret_cast<uint8_t const*>(array.data());
		size_t const byteLength = array.size() * sizeof(T);
		for (size_t i = 0; i != byteLength; ++i) {
			CHECK(byteArray[i] == 0);
		}
	}
}

T_DECL(ctor_allocate_zero, "safe_allocation.ctor.allocate_zero", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
	tests<int>();
	tests<int const>();
}
