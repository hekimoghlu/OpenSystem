/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
// TEST_ENV OBJC_DEBUG_POOL_DEPTH=-1

#include "test.h"
#include <objc/objc-exception.h>
#include <objc/NSObject.h>

static int state;

@interface Foo : NSObject @end
@interface Bar : NSObject @end

@interface Foo (Unimplemented)
+(void)method;
@end

@implementation Bar @end

@implementation Foo

-(void)check { state++; }
+(void)check { testassert(!"caught class object, not instance"); }

static id exc;

static void handler(id unused, void *ctx) __attribute__((used));
static void handler(id unused __unused, void *ctx __unused)
{
    testassert(state == 3); state++;
}

+(BOOL) resolveClassMethod:(SEL)__unused name
{
    testassertequal(state, 1); state++;
#if TARGET_OS_EXCLAVEKIT
    state++;  // handler would have done this
#elif TARGET_OS_OSX
    objc_addExceptionHandler(&handler, 0);
    testassertequal(state, 2); 
#else
    state++;  // handler would have done this
#endif
    state++;
    exc = [Foo new];
    @throw exc;
}


@end

int main()
{
    // unwind exception and alt handler through objc_msgSend()

    PUSH_POOL {

#if TARGET_OS_EXCLAVEKIT
        const int count = 256;
#else
        const int count = is_guardmalloc() ? 1000 : 100000;
#endif
        state = 0;
        for (int i = 0; i < count; i++) {
            @try {
                testassertequal(state, 0); state++;
                [Foo method];
                testunreachable();
            } @catch (Bar *e) {
                testunreachable();
            } @catch (Foo *e) {
                testassertequal(e, exc);
                testassertequal(state, 4); state++;
                testassertequal(state, 5); [e check];  // state++
                RELEASE_VAR(exc);
            } @catch (id e) {
                testunreachable();
            } @catch (...) {
                testunreachable();
            } @finally {
                testassertequal(state, 6); state++;
            }
            testassertequal(state, 7); state = 0;
        }

    } POP_POOL;

    succeed(__FILE__);
}
