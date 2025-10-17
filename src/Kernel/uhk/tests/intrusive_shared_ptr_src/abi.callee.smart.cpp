/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
// Declare a function as returning a raw pointer (in the header), but
// implement it by returning a shared pointer. This represents a TU that
// would have been translated to shared pointers.
//
// In this TU, SharedPtr<T> is intrusive_shared_ptr<T>, since USE_SHARED_PTR
// is defined.
//

#define USE_SHARED_PTR

#include "abi_helper.h"

SharedPtr<T>
return_shared_as_raw(T* ptr)
{
	return SharedPtr<T>(ptr, libkern::no_retain);
}
