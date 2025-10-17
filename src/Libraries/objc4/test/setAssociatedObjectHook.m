/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#include "testroot.i"

bool hasAssociations = false;

@interface TestRoot (AssocHooks)
@end

@implementation TestRoot (AssocHooks)

- (void)_noteAssociatedObjects {
  hasAssociations = true;
}

// -_noteAssociatedObjects is currently limited to raw-isa custom-rr to avoid overhead
- (void) release {
}

@end

int main() {
    // Intel simulator doesn't support this method.
#if !TARGET_OS_SIMULATOR || !__x86_64__
    id obj = [TestRoot new];
    id value = [TestRoot new];
    const void *key = "key";
    objc_setAssociatedObject(obj, key, value, OBJC_ASSOCIATION_RETAIN);
    testassert(hasAssociations == true);

    id out = objc_getAssociatedObject(obj, key);
    testassert(out == value);

    hasAssociations = false;
    key = "key2";
    objc_setAssociatedObject(obj, key, value, OBJC_ASSOCIATION_RETAIN);
    testassert(hasAssociations == false); //only called once


    out = objc_getAssociatedObject(obj, key);
    testassert(out == value);
#endif

    succeed(__FILE__);
}
