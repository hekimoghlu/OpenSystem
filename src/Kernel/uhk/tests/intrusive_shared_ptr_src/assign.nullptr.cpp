/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
//  intrusive_shared_ptr& operator=(std::nullptr_t);
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

	// Assign nullptr to non-null
	{
		tracked_shared_ptr<T> ptr(&obj, libkern::retain);
		tracking_policy::reset();
		tracked_shared_ptr<T>& ref = (ptr = nullptr);
		CHECK(tracking_policy::releases == 1);
		CHECK(tracking_policy::retains == 0);
		CHECK(&ref == &ptr);
		CHECK(ptr.get() == nullptr);
	}

	// Assign nullptr to null
	{
		tracked_shared_ptr<T> ptr = nullptr;
		tracking_policy::reset();
		tracked_shared_ptr<T>& ref = (ptr = nullptr);
		CHECK(tracking_policy::releases == 0);
		CHECK(tracking_policy::retains == 0);
		CHECK(&ref == &ptr);
		CHECK(ptr.get() == nullptr);
	}
}

T_DECL(assign_nullptr, "intrusive_shared_ptr.assign.nullptr", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
