/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
/*!
    @header SecInternal
    SecInternal defines common internal constants macros and SPI functions.
*/

#ifndef _SECURITY_SECINTERNAL_H_
#define _SECURITY_SECINTERNAL_H_

#include "utilities/simulatecrash_assert.h"
#include <CoreFoundation/CFNumber.h>
#include <CoreFoundation/CFString.h>

#include <Security/SecBase.h>

__BEGIN_DECLS

#include "utilities/SecCFRelease.h"

#define AssignOrReleaseResult(CF,OUT) { \
     CFTypeRef _cf = (CF), *_out = (OUT); \
     if (_out) { *_out = _cf; } else { if (_cf) CFRelease(_cf); } }

#define DICT_DECLARE(MAXVALUES) \
     CFIndex numValues = 0, maxValues = (MAXVALUES); \
     const void *keys[maxValues]; \
     const void *values[maxValues];

#define DICT_ADDPAIR(KEY,VALUE) do { \
     if (numValues < maxValues) { \
          keys[numValues] = (KEY); \
          values[numValues] = (VALUE); \
          numValues++; \
     } else \
          assert(false); \
} while(0)

#define DICT_CREATE(ALLOCATOR) CFDictionaryCreate((ALLOCATOR), keys, values, \
     numValues, NULL, &kCFTypeDictionaryValueCallBacks)

/* Non valid CFTimeInterval or CFAbsoluteTime. */
#define NULL_TIME    0.0

#if SEC_OS_IPHONE
static inline CFIndex getIntValue(CFTypeRef cf) {
    if (cf) {
        if (CFGetTypeID(cf) == CFNumberGetTypeID()) {
            CFIndex value;
            CFNumberGetValue(cf, kCFNumberCFIndexType, &value);
            return value;
        } else if (CFGetTypeID(cf) == CFStringGetTypeID()) {
            return CFStringGetIntValue(cf);
        }
    }
    return -1;
}
#endif // SEC_OS_IPHONE

__END_DECLS

#endif /* !_SECURITY_SECINTERNAL_H_ */
