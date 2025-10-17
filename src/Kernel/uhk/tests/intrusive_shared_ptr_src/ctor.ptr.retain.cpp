/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
//  explicit intrusive_shared_ptr(pointer p, retain_t);
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include "test_policy.h"

struct T { int i; };

template <typename T, typename TQual>
static void
tests()
{
	T obj{0};

	{
		test_policy::retain_count = 0;
		libkern::intrusive_shared_ptr<TQual, test_policy> ptr(&obj, libkern::retain);
		CHECK(ptr.get() == &obj);
		CHECK(test_policy::retain_count == 1);
	}
	{
		test_policy::retain_count = 0;
		libkern::intrusive_shared_ptr<TQual, test_policy> ptr{&obj, libkern::retain};
		CHECK(ptr.get() == &obj);
		CHECK(test_policy::retain_count == 1);
	}
}

T_DECL(ctor_ptr_retain, "intrusive_shared_ptr.ctor.ptr.retain", T_META_TAG_VM_PREFERRED) {
	tests<T, T>();
	tests<T, T const>();
}
