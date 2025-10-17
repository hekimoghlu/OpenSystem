/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
 * Copyright (c) 2002-2006, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************
 *
 * @author Mark E. Davis
 * @author Vladimir Weinstein
 */

/**
 * Test Canonical Iterator
 */

#ifndef _CANITTST
#define _CANITTST

#include "unicode/utypes.h"

#if !UCONFIG_NO_NORMALIZATION


U_NAMESPACE_BEGIN

class Transliterator;

U_NAMESPACE_END

#include "unicode/translit.h"
#include "unicode/caniter.h"
#include "intltest.h"
#include "hash.h"

class CanonicalIteratorTest : public IntlTest {
public:
    CanonicalIteratorTest();
    virtual ~CanonicalIteratorTest();

    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    void TestCanonicalIterator();
    void TestExhaustive();
    void TestBasic();
    void TestAPI();
    UnicodeString collectionToString(Hashtable *col);
    //static UnicodeString collectionToString(Collection col);
private:
    void expectEqual(const UnicodeString &message, const UnicodeString &item, const UnicodeString &a, const UnicodeString &b);
    void characterTest(UnicodeString &s, UChar32 ch, CanonicalIterator &it);

    Transliterator *nameTrans;
    Transliterator *hexTrans;
        
    UnicodeString getReadable(const UnicodeString &obj);
};

#endif /* #if !UCONFIG_NO_NORMALIZATION */

#endif // _CANITTST
