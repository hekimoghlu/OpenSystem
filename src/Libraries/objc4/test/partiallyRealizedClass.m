/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#include <objc/NSObject.h>
#include "test.h"

@interface AccessStatic: NSObject @end
@implementation AccessStatic @end

// EXTERN_C id objc_retainAutoreleaseReturnValue(id);
// EXTERN_C id objc_alloc(id);

// Verify that objc_retainAutoreleaseReturnValue on an unrealized class doesn't
// put the class into a half-baked state where the metaclass is realized but the
// class is not. rdar://101151980
int main() {
    extern char OBJC_CLASS_$_AccessStatic;
    id st = (__bridge id)(void *)&OBJC_CLASS_$_AccessStatic;

    objc_retainAutoreleaseReturnValue(st);
    objc_alloc(st);

    succeed(__FILE__);
}
