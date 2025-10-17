/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include <objc/objc-runtime.h>

@interface Sub : TestRoot @end
@implementation Sub @end

int main()
{
    // [super ...] messages are tested in msgSend.m

    testassert(class_getSuperclass([Sub class]) == [TestRoot class]);
    testassert(class_getSuperclass(object_getClass([Sub class])) == object_getClass([TestRoot class]));
    testassert(class_getSuperclass([TestRoot class]) == Nil);
    testassert(class_getSuperclass(object_getClass([TestRoot class])) == [TestRoot class]);
    testassert(class_getSuperclass(Nil) == Nil);

    succeed(__FILE__);
}
