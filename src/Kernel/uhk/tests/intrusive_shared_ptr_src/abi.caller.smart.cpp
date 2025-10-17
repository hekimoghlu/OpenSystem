/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
// This tests that we can call functions implemented using raw pointers from
// an API vending itself as returning shared pointers, because both are ABI
// compatible.
//
// In this TU, SharedPtr<T> is intrusive_shared_ptr<T>, since USE_SHARED_PTR
// is defined.
//

#define USE_SHARED_PTR

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "abi_helper.h"

static_assert(sizeof(SharedPtr<T>) == sizeof(T*));
static_assert(alignof(SharedPtr<T>) == alignof(T*));

// Receive a shared pointer from a function that actually returns a raw pointer
T_DECL(abi_caller_smart, "intrusive_shared_ptr.abi.caller.smart", T_META_TAG_VM_PREFERRED) {
	T obj{3};
	T* expected = &obj;
	SharedPtr<T> result = return_raw_as_shared(expected);
	CHECK(result.get() == expected);

	// Sometimes the test above passes even though it should fail, if the
	// right address happens to be on the stack in the right location. This
	// can happen if abi.caller.raw is run just before this test. This second
	// test makes sure it fails when it should.
	CHECK(result->i == 3);
}
