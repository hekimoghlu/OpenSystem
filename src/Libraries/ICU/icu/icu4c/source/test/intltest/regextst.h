/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
 * Copyright (c) 2002-2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/


#ifndef REGEXTST_H
#define REGEXTST_H

#include "unicode/utypes.h"
#if !UCONFIG_NO_REGULAR_EXPRESSIONS

#include "intltest.h"

struct UText;
typedef struct UText UText;

class RegexTest: public IntlTest {
public:

    RegexTest();
    virtual ~RegexTest();

    virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    // The following are test functions that are visible from the intltest test framework.
    virtual void API_Match();
    virtual void API_Pattern();
    virtual void API_Replace();
    virtual void Basic();
    virtual void Extended();
    virtual void Errors();
    virtual void PerlTests();
    virtual void Bug6149();
    virtual void Callbacks();
    virtual void FindProgressCallbacks();
    virtual void UTextBasic();
    virtual void API_Match_UTF8();
    virtual void API_Pattern_UTF8();
    virtual void API_Replace_UTF8();
    virtual void PerlTestsUTF8();
    virtual void PreAllocatedUTextCAPI();
    virtual void NamedCapture();
    virtual void NamedCaptureLimits();
    virtual void Bug7651();
    virtual void Bug7740();
    virtual void Bug8479();
    virtual void Bug7029();
    virtual void Bug9283();
    virtual void CheckInvBufSize();
    virtual void Bug10459();
    virtual void TestCaseInsensitiveStarters();
    virtual void TestBug11049();
    virtual void TestBug11371();
    virtual void TestBug11480();
    virtual void TestBug12884();
    virtual void TestBug13631();
    virtual void TestBug13632();
    virtual void TestBug20359();
    virtual void TestBug20863();
#if APPLE_ICU_CHANGES
    virtual void TestForHang(); // rdar://131705700 (see also: ICU-23047)
#endif // APPLE_ICU_CHANGES

    // The following functions are internal to the regexp tests.
    virtual void assertUText(const char *expected, UText *actual, const char *file, int line);
    virtual void assertUTextInvariant(const char *invariant, UText *actual, const char *file, int line);
    virtual UBool doRegexLMTest(const char *pat, const char *text, UBool looking, UBool match, int32_t line);
    virtual UBool doRegexLMTestUTF8(const char *pat, const char *text, UBool looking, UBool match, int32_t line);
    virtual void regex_find(const UnicodeString &pat, const UnicodeString &flags,
                            const UnicodeString &input, const char *srcPath, int32_t line);
    virtual void regex_err(const char *pat, int32_t errline, int32_t errcol,
                            UErrorCode expectedStatus, int32_t line);
    virtual const char *getPath(char buffer[2048], const char *filename);

    virtual void TestCase11049(const char *pattern, const char *data, UBool expectMatch, int32_t lineNumber);

    static const char* extractToAssertBuf(const UnicodeString& message);
    
};

#endif   // !UCONFIG_NO_REGULAR_EXPRESSIONS
#endif
