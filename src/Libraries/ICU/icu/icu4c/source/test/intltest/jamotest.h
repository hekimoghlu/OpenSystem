/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
 ********************************************************************
************************************************************************
*   Date        Name        Description
*   02/28/2001  aliu        Creation
*   03/01/2001  George      port to HP/UX
************************************************************************/

#ifndef JAMOTEST_H
#define JAMOTEST_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_TRANSLITERATION

#include "unicode/translit.h"
#include "transtst.h"

/**
 * @test
 * @summary Test of Latin-Jamo and Jamo-Latin rules
 */
class JamoTest : public TransliteratorTest {
public:
    JamoTest();
    virtual ~JamoTest();
private:
    void runIndexedTest(int32_t index, UBool exec, const char* &name,
                        char* par=nullptr) override;

    void TestJamo();
    
    void TestRealText();

    void TestPiecemeal();

    //======================================================================
    // Support methods
    //======================================================================

    // Override TransliteratorTest
    virtual void expectAux(const UnicodeString& tag,
                           const UnicodeString& summary, UBool pass,
                           const UnicodeString& expectedResult) override;

    // Methods to convert Jamo to/from readable short names,
    // e.g. (Gi) <> U+1100
    static const char* JAMO_NAMES_RULES;
    Transliterator *JAMO_NAME;
    Transliterator *NAME_JAMO;
    UnicodeString nameToJamo(const UnicodeString& input);
    UnicodeString jamoToName(const UnicodeString& input);
};

#endif /* #if !UCONFIG_NO_TRANSLITERATION */

#endif
