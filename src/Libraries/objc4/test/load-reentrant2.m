/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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

int state2 = 0;
extern int state1;

static void ctor(void)  __attribute__((constructor));
static void ctor(void) 
{
    // should be called during One's dlopen(), before Two's +load
    testassert(state1 == 111);
    testassert(state2 == 0);
}

OBJC_ROOT_CLASS
@interface Two @end
@implementation Two
+(void) load
{
    // Does not run until One's +load completes
    testassert(state1 == 1);
    state2 = 2;
}
@end
