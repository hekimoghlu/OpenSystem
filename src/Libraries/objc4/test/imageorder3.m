/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

static void c3(void) __attribute__((constructor));
static void c3(void)
{
    testassert(state == 3);  // +load before C/C++
    testassert(cstate == 2);
    cstate = 3;
}


#if __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
#endif

@implementation Super (cat3)
+(void) method {
    state = 3;
}
+(void) method3 {
    state = 3;
}
+(void) load {
    testassert(state == 2);
    state = 3;
}
@end

#if __clang__
#pragma clang diagnostic pop
#endif
