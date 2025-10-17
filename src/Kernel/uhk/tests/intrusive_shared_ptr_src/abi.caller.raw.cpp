/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
// This tests that we can call functions implemented using shared pointers
// from an API vending itself as returning raw pointers, because both are
// ABI compatible.
//
// In this TU, SharedPtr<T> is just T*, since USE_SHARED_PTR is not defined.
//

#include <darwintest.h>
#include "abi_helper.h"

// Receive a raw pointer from a function that actually returns a smart pointer
T_DECL(abi_caller_raw, "intrusive_shared_ptr.abi.caller.raw", T_META_TAG_VM_PREFERRED) {
	T obj{10};
	T* expected = &obj;
	T* result = return_shared_as_raw(expected);
	CHECK(result == expected);

	// Sometimes the test above passes even though it should fail, if the
	// right address happens to be on the stack in the right location. This
	// can happen if abi.caller.smart is run just before this test. This
	// second test makes sure it fails when it should.
	CHECK(result->i == 10);
}
