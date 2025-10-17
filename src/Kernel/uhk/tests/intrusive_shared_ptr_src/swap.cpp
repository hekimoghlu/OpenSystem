/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
//  void swap(intrusive_shared_ptr& a, intrusive_shared_ptr& b);
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct T { int i; };

template <typename T>
static void
tests()
{
	T obj1{1};
	T obj2{2};

	// Swap non-null with non-null
	{
		tracked_shared_ptr<T> a(&obj1, libkern::retain);
		tracked_shared_ptr<T> b(&obj2, libkern::retain);
		T* a_raw = a.get();
		T* b_raw = b.get();
		tracking_policy::reset();

		swap(a, b); // ADL call

		CHECK(tracking_policy::retains == 0);
		CHECK(tracking_policy::releases == 0);
		CHECK(a.get() == b_raw);
		CHECK(b.get() == a_raw);
	}

	// Swap non-null with null
	{
		tracked_shared_ptr<T> a(&obj1, libkern::retain);
		tracked_shared_ptr<T> b = nullptr;
		T* a_raw = a.get();
		tracking_policy::reset();

		swap(a, b); // ADL call

		CHECK(tracking_policy::retains == 0);
		CHECK(tracking_policy::releases == 0);
		CHECK(a.get() == nullptr);
		CHECK(b.get() == a_raw);
	}

	// Swap null with non-null
	{
		tracked_shared_ptr<T> a = nullptr;
		tracked_shared_ptr<T> b(&obj2, libkern::retain);
		T* b_raw = b.get();
		tracking_policy::reset();

		swap(a, b); // ADL call

		CHECK(tracking_policy::retains == 0);
		CHECK(tracking_policy::releases == 0);
		CHECK(a.get() == b_raw);
		CHECK(b.get() == nullptr);
	}

	// Swap null with null
	{
		tracked_shared_ptr<T> a = nullptr;
		tracked_shared_ptr<T> b = nullptr;
		tracking_policy::reset();

		swap(a, b); // ADL call

		CHECK(tracking_policy::retains == 0);
		CHECK(tracking_policy::releases == 0);
		CHECK(a.get() == nullptr);
		CHECK(b.get() == nullptr);
	}

	// Swap with self
	{
		tracked_shared_ptr<T> a(&obj1, libkern::retain);
		T* a_raw = a.get();
		tracking_policy::reset();

		swap(a, a); // ADL call

		CHECK(tracking_policy::retains == 0);
		CHECK(tracking_policy::releases == 0);
		CHECK(a.get() == a_raw);
	}
}

T_DECL(swap, "intrusive_shared_ptr.swap", T_META_TAG_VM_PREFERRED) {
	tests<T>();
	tests<T const>();
}
