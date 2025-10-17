/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#ifndef TESTS_INTRUSIVE_SHARED_PTR_ABI_HELPER_H
#define TESTS_INTRUSIVE_SHARED_PTR_ABI_HELPER_H

#include <libkern/c++/intrusive_shared_ptr.h>
#include <darwintest.h>
#include "test_policy.h"

struct T { int i; };

#if defined USE_SHARED_PTR
template <typename T>
using SharedPtr = libkern::intrusive_shared_ptr<T, test_policy>;
#else
template <typename T>
using SharedPtr = T *;
#endif

extern SharedPtr<T> return_shared_as_raw(T*);
extern SharedPtr<T> return_raw_as_shared(T*);

#endif // !TESTS_INTRUSIVE_SHARED_PTR_ABI_HELPER_H
