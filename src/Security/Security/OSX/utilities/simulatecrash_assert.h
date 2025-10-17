/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#ifndef _SECURITY_UTILITIES_SIMULATECRASH_ASSERT_H_
#define _SECURITY_UTILITIES_SIMULATECRASH_ASSERT_H_

#if !defined(NDEBUG) || !defined(__OBJC__)
    #include <assert.h>
#else // NDEBUG && __OBJC__
    #include <security_utilities/debugging.h>
    #undef assert
    #define assert(expr) { \
        if (!(expr)) { \
            __security_simulatecrash(CFSTR("Execution has encountered an unexpected state"), __sec_exception_code_UnexpectedState); \
        } \
    }
#endif // NDEBUG && __OBJC__

#endif /* _SECURITY_UTILITIES_SIMULATECRASH_ASSERT_H_ */
