/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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

// TEST_CFLAGS -funsigned-char
// TEST_CONFIG LANGUAGE=c,c++,objc,objc++
// (verify -funsigned-char doesn't change the definition of BOOL)

#include "test.h"
#include <objc/objc.h>

#if TARGET_OS_OSX
#   if __x86_64__
#       define RealBool 0
#   else
#       define RealBool 1
#   endif
#elif TARGET_OS_IOS || TARGET_OS_BRIDGE
#   if (__arm__ && !__armv7k__) || __i386__
#       define RealBool 0
#   else
#       define RealBool 1
#   endif
#else
#   define RealBool 1
#endif

#if __OBJC__ && !defined(__OBJC_BOOL_IS_BOOL)
#   error no __OBJC_BOOL_IS_BOOL
#endif

#if RealBool != OBJC_BOOL_IS_BOOL
#   error wrong OBJC_BOOL_IS_BOOL
#endif

#if RealBool == OBJC_BOOL_IS_CHAR
#   error wrong OBJC_BOOL_IS_CHAR
#endif

int main()
{
    const char *expected __unused =
#if RealBool
        "B"
#else
        "c"
#endif
        ;
#if __OBJC__
    const char *enc = @encode(BOOL);
    testassert(0 == strcmp(enc, expected));
#endif
    succeed(__FILE__);
}
