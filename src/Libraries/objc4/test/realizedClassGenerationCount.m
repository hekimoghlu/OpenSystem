/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

// TEST_CONFIG

#include "test.h"
#include "testroot.i"

#include <dlfcn.h>

extern uintptr_t objc_debug_realized_class_generation_count;

int main()
{
    testassert(objc_debug_realized_class_generation_count > 0);
    uintptr_t prev = objc_debug_realized_class_generation_count;

    Class c;

#if !TARGET_OS_EXCLAVEKIT
    void *handle = dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", RTLD_LAZY);
    testassert(handle);
    c = objc_getClass("NSFileManager");
    testassert(c);
    testassert(objc_debug_realized_class_generation_count > prev);
#endif

    prev = objc_debug_realized_class_generation_count;
    c = objc_allocateClassPair([TestRoot class], "Dynamic", 0);
    testassert(objc_debug_realized_class_generation_count > prev);
    prev = objc_debug_realized_class_generation_count;
    objc_registerClassPair(c);
    testassert(objc_debug_realized_class_generation_count == prev);
    
    succeed(__FILE__);
}
