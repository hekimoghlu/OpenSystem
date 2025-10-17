/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
//  T& operator*() const noexcept;
//  T* operator->() const noexcept;
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct T {
	int i;
};

template <typename T>
static void
tests()
{
	T obj{3};
	tracked_shared_ptr<T> ptr(&obj, libkern::no_retain);

	{
		T& ref = *ptr;
		CHECK(&ref == &obj);
		CHECK(ref.i == 3);
	}

	{
		int const& ref = ptr->i;
		CHECK(&ref == &obj.i);
		CHECK(ref == 3);
	}
}

T_DECL(deref, "intrusive_shared_ptr.deref", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
