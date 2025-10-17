/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
*******************************************************************************
*   Copyright (C) 2011-2012, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:  appendable.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2010dec07
*   created by: Markus W. Scherer
*/

#include "unicode/utypes.h"
#include "unicode/appendable.h"
#include "unicode/utf16.h"

U_NAMESPACE_BEGIN

Appendable::~Appendable() {}

UBool
Appendable::appendCodePoint(UChar32 c) {
    if(c<=0xffff) {
        return appendCodeUnit(static_cast<char16_t>(c));
    } else {
        return appendCodeUnit(U16_LEAD(c)) && appendCodeUnit(U16_TRAIL(c));
    }
}

UBool
Appendable::appendString(const char16_t *s, int32_t length) {
    if(length<0) {
        char16_t c;
        while((c=*s++)!=0) {
            if(!appendCodeUnit(c)) {
                return false;
            }
        }
    } else if(length>0) {
        const char16_t *limit=s+length;
        do {
            if(!appendCodeUnit(*s++)) {
                return false;
            }
        } while(s<limit);
    }
    return true;
}

UBool
Appendable::reserveAppendCapacity(int32_t /*appendCapacity*/) {
    return true;
}

char16_t *
Appendable::getAppendBuffer(int32_t minCapacity,
                            int32_t /*desiredCapacityHint*/,
                            char16_t *scratch, int32_t scratchCapacity,
                            int32_t *resultCapacity) {
    if(minCapacity<1 || scratchCapacity<minCapacity) {
        *resultCapacity=0;
        return nullptr;
    }
    *resultCapacity=scratchCapacity;
    return scratch;
}

// UnicodeStringAppendable is implemented in unistr.cpp.

U_NAMESPACE_END
