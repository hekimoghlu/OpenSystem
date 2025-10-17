/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

#include <objc/objc-exception.h>

/*
  rdar://6888838
  Mail installs an alt handler on one thread and deletes it on another.
  This confuses the alt handler machinery, which halts the process.
*/

uintptr_t Token;

void handler(id unused __unused, void *context __unused)
{
}

int main()
{
#if __clang__ && __cplusplus
    // alt handlers need the objc personality
    // catch (id) workaround forces the objc personality
    @try {
        testwarn("rdar://9183014 clang uses wrong exception personality");
    } @catch (id e __unused) {
    }
#endif

    @try {
        // Install 4 alt handlers
        uintptr_t t1, t2, t3, t4;
        t1 = objc_addExceptionHandler(&handler, NULL);
        t2 = objc_addExceptionHandler(&handler, NULL);
        t3 = objc_addExceptionHandler(&handler, NULL);
        t4 = objc_addExceptionHandler(&handler, NULL);

        // Remove 3 of them.
        objc_removeExceptionHandler(t1);
        objc_removeExceptionHandler(t2);
        objc_removeExceptionHandler(t3);
        
        // Create an alt handler on another thread 
        // that collides with one of the removed handlers
        testonthread(^{
            @try {
                Token = objc_addExceptionHandler(&handler, NULL);
            } @catch (...) {
            }
        });
        
        // Incorrectly remove the other thread's handler
        objc_removeExceptionHandler(Token);
        // Remove the 4th handler
        objc_removeExceptionHandler(t4);
        
        // Install 8 more handlers.
        // If the other thread's handler was not ignored, 
        // this will fail.
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
        objc_addExceptionHandler(&handler, NULL);
    } @catch (...) {
    }

    // This should have crashed earlier.
    fail(__FILE__);
}
