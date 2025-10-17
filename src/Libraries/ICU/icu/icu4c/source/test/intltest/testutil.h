/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
**********************************************************************
*   Copyright (C) 2001-2009, International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*   Date        Name        Description
*   05/23/00    aliu        Creation.
**********************************************************************
*/
#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "unicode/utypes.h"
#include "unicode/edits.h"
#include "unicode/unistr.h"
#include "intltest.h"

struct EditChange {
    UBool change;
    int32_t oldLength, newLength;
};

/**
 * Utility methods. Everything in this class is static.
 */
class TestUtility {
public:
    static UnicodeString &appendHex(UnicodeString &buf, UChar32 ch);

    static UnicodeString hex(UChar32 ch);

    static UnicodeString hex(const UnicodeString& s);

    static UnicodeString hex(const UnicodeString& s, char16_t sep);

    static UnicodeString hex(const uint8_t* bytes, int32_t len);

    static UBool checkEqualEdits(IntlTest &test, const UnicodeString &name,
                                 const Edits &e1, const Edits &e2, UErrorCode &errorCode);

    static void checkEditsIter(
        IntlTest &test, const UnicodeString &name,
        Edits::Iterator ei1, Edits::Iterator ei2,  // two equal iterators
        const EditChange expected[], int32_t expLength, UBool withUnchanged,
        UErrorCode &errorCode);

private:
    TestUtility() = delete;  // Prevent instantiation
};

#endif
