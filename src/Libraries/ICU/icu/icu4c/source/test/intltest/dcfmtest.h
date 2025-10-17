/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
 * Copyright (c) 2010-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

//
//   file:  dcfmtest.h
//
//   Data driven decimal formatter test.
//      Includes testing of both parsing and formatting.
//      Tests are in the text file dcfmtest.txt, in the source/test/testdata/ directory.
//

#ifndef DCFMTEST_H
#define DCFMTEST_H

#include "unicode/utypes.h"
#if !UCONFIG_NO_REGULAR_EXPRESSIONS

#include "intltest.h"


class DecimalFormatTest: public IntlTest {
public:

    DecimalFormatTest();
    virtual ~DecimalFormatTest();

    virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    // The following are test functions that are visible from the intltest test framework.
    virtual void DataDrivenTests();

    virtual const char *getPath(char buffer[2048], const char *filename);
    virtual void execParseTest(int32_t lineNum,
                              const UnicodeString &inputText,
                              const UnicodeString &expectedType,
                              const UnicodeString &expectedDecimal,
                              UErrorCode &status);

private:
    enum EFormatInputType {
        kFormattable,
        kStringPiece
    };

public:
    virtual void execFormatTest(int32_t lineNum,
                               const UnicodeString &pattern,
                               const UnicodeString &round, 
                               const UnicodeString &input,
                               const UnicodeString &expected,
                               EFormatInputType inType,
                               UErrorCode &status);
};

#endif   // !UCONFIG_NO_REGULAR_EXPRESSIONS
#endif
