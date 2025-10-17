/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

// TEST_CONFIG MEM=arc LANGUAGE=objective-c OS=!exclavekit
// TEST_ENV OBJC_DEBUG_SCRIBBLE_CACHES=YES
// TEST_NO_MALLOC_SCRIBBLE

// Stress test thread-safe cache deallocation and reallocation.

#include "test.h"
#include "testroot.i"
#include <dispatch/dispatch.h>

@interface MyClass1 : TestRoot
@end
@implementation MyClass1
@end

@interface MyClass2 : TestRoot
@end
@implementation MyClass2
@end

@interface MyClass3 : TestRoot
@end
@implementation MyClass3
@end

@interface MyClass4 : TestRoot
@end
@implementation MyClass4
@end

int main() {
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        usleep(200000);
        while (1) {
            usleep(1000);
            _objc_flush_caches(MyClass1.class);
            _objc_flush_caches(MyClass2.class);
            _objc_flush_caches(MyClass3.class);
            _objc_flush_caches(MyClass4.class);
        }
    });
    
    for (int i = 0; i < 6; i++) {
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            long j = 0;
            while (1) {
                j++;
                (void)[[MyClass1 alloc] init];
                (void)[[MyClass2 alloc] init];
                (void)[[MyClass3 alloc] init];
                (void)[[MyClass4 alloc] init];
            }
        });
    }
    
    sleep(5);
    
    succeed(__FILE__);
    
    return 0;
}
