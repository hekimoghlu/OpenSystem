/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
 *
 *   Copyright (C) 2005, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *
 *******************************************************************************
 *
 *   created on: 2005jun15
 *   created by: Raymond Yang
 */

#ifndef IDNA_CONF_TEST_H
#define IDNA_CONF_TEST_H

#include "intltest.h"
#include "unicode/ustring.h"


class IdnaConfTest: public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par=nullptr) override;
    IdnaConfTest();
    virtual ~IdnaConfTest();
private:
    void Test();

    // for test file handling
    char16_t* base;
    int len ;
    int curOffset;

    int isNewlineMark();
    UBool ReadOneLine(UnicodeString&);

    // for parsing one test record
    UnicodeString id;   // for debug & error output
    UnicodeString namebase;
    UnicodeString namezone;
    int type;     // 0 toascii,             1 tounicode
    int option;   // 0 UseSTD3ASCIIRules,   1 ALLOW_UNASSIGNED
    int passfail; // 0 pass,                1 fail

    void ExplainCodePointTag(UnicodeString& buf);
    void Call();
};

#endif /*IDNA_CONF_TEST_H*/
