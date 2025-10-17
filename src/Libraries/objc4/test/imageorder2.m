/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "imageorder.h"

static void c2(void) __attribute__((constructor));
static void c2(void)
{
    testassert(state == 2);  // +load before C/C++
    testassert(cstate == 1);
    cstate = 2;
}


#if __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
#endif

@implementation Super (cat2)
+(void) method {
    fail("+[Super(cat2) method] not replaced!");
}
+(void) method2 {
    state = 2;
}
+(void) load {
    testassert(state == 1);
    state = 2;
}
@end

#if __clang__
#pragma clang diagnostic pop
#endif
