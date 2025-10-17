/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

// TEST_CONFIG ARCH=x86_64 MEM=mrc
// TEST_CFLAGS -framework Foundation

// rdar://20206767

#include <Foundation/Foundation.h>
#include "test.h"


@interface Test : NSObject @end
@implementation Test
@end


int main()
{
    id buf[1];
    buf[0] = [Test class];
    id obj = (id)buf;
    [obj retain];
    [obj retain];

    uintptr_t rax;

    [obj release];
    asm("mov %%rax, %0" : "=r" (rax));
    testassert(rax == 0);

    objc_release(obj);
    asm("mov %%rax, %0" : "=r" (rax));
    testassert(rax == 0);

    succeed(__FILE__);
}
