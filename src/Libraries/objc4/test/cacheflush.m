/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#include "test.h"
#include <objc/runtime.h>
#include <dlfcn.h>

#include "cacheflush.h"

@interface Sub : TestRoot @end
@implementation Sub @end


int main()
{
    TestRoot *sup = [TestRoot new];
    Sub *sub = [Sub new];

    // Fill method cache
    testassert(1 == [TestRoot classMethod]);
    testassert(1 == [sup instanceMethod]);
    testassert(1 == [TestRoot classMethod]);
    testassert(1 == [sup instanceMethod]);

    testassert(1 == [Sub classMethod]);
    testassert(1 == [sub instanceMethod]);
    testassert(1 == [Sub classMethod]);
    testassert(1 == [sub instanceMethod]);

#if !TARGET_OS_EXCLAVEKIT
    // Dynamically load a category
    dlopen("cacheflush2.dylib", 0);

    // Make sure old cache results are gone
    testassert(2 == [TestRoot classMethod]);
    testassert(2 == [sup instanceMethod]);

    testassert(2 == [Sub classMethod]);
    testassert(2 == [sub instanceMethod]);

    // Dynamically load another category
    dlopen("cacheflush3.dylib", 0);

    // Make sure old cache results are gone
    testassert(3 == [TestRoot classMethod]);
    testassert(3 == [sup instanceMethod]);

    testassert(3 == [Sub classMethod]);
    testassert(3 == [sub instanceMethod]);
#endif // !TARGET_OS_EXCLAVEKIT

    // fixme test subclasses

    // fixme test objc_flush_caches(), class_addMethod(), class_addMethods()

    succeed(__FILE__);
}
