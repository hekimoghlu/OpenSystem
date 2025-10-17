/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
/********************************************************************
 * COPYRIGHT: 
 * Copyright (c) 1997-2001, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * MajorTestLevel is the top level test class for everything in the directory "IntlWork".
 */

#ifndef _INTLTESTFORMAT
#define _INTLTESTFORMAT

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/formattedvalue.h"
#include "intltest.h"


class IntlTestFormat: public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
};


typedef struct UFieldPositionWithCategory {
    UFieldCategory category;
    int32_t field;
    int32_t beginIndex;
    int32_t endIndex;
} UFieldPositionWithCategory;

class IntlTestWithFieldPosition : public IntlTest {
public:
    // Tests FormattedValue's toString, toTempString, and nextPosition methods.
    //
    // expectedCategory gets combined with expectedFieldPositions to call
    // checkMixedFormattedValue.
    void checkFormattedValue(
        const char16_t* message,
        const FormattedValue& fv,
        UnicodeString expectedString,
        UFieldCategory expectedCategory,
        const UFieldPosition* expectedFieldPositions,
        int32_t length);

    // Tests FormattedValue's toString, toTempString, and nextPosition methods.
    void checkMixedFormattedValue(
        const char16_t* message,
        const FormattedValue& fv,
        UnicodeString expectedString,
        const UFieldPositionWithCategory* expectedFieldPositions,
        int32_t length);
};


#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
