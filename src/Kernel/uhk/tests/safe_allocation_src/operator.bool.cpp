/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
//      explicit operator bool() const;
//

#include <libkern/c++/safe_allocation.h>
#include <darwintest.h>
#include "test_utils.h"
#include <type_traits>

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	{
		test_safe_allocation<T> const array(10, libkern::allocate_memory);
		CHECK(static_cast<bool>(array));
		if (array) {
		} else {
			CHECK(FALSE);
		}
	}
	{
		test_safe_allocation<T> const array = nullptr;
		CHECK(!static_cast<bool>(array));
		if (!array) {
		} else {
			CHECK(FALSE);
		}
	}

	static_assert(!std::is_convertible_v<test_safe_allocation<T>, bool>);
}

T_DECL(operator_bool, "safe_allocation.operator.bool", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
