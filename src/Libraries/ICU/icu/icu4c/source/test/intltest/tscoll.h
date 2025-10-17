/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
 * Copyright (c) 1997-2008, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * MajorTestLevel is the top level test class for everything in the directory "IntlWork".
 */

#ifndef _INTLTESTCOLLATOR
#define _INTLTESTCOLLATOR

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "intltest.h"
#include "unicode/coll.h"
#include "unicode/coleitr.h"


class IntlTestCollator: public IntlTest {
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
protected:
    struct Order
    {
        int32_t order;
        int32_t offset;
    };

    // These two should probably go down in IntlTest
    void doTest(Collator* col, const char16_t *source, const char16_t *target, Collator::EComparisonResult result);

    void doTest(Collator* col, const UnicodeString &source, const UnicodeString &target, Collator::EComparisonResult result);
    void doTestVariant(Collator* col, const UnicodeString &source, const UnicodeString &target, Collator::EComparisonResult result);
    virtual void reportCResult( const UnicodeString &source, const UnicodeString &target,
                                CollationKey &sourceKey, CollationKey &targetKey,
                                Collator::EComparisonResult compareResult,
                                Collator::EComparisonResult keyResult,
                                Collator::EComparisonResult incResult,
                                Collator::EComparisonResult expectedResult );

    static UnicodeString &prettify(const CollationKey &source, UnicodeString &target);
    static UnicodeString &appendCompareResult(Collator::EComparisonResult result, UnicodeString &target);
    void backAndForth(CollationElementIterator &iter);
    /**
     * Return an integer array containing all of the collation orders
     * returned by calls to next on the specified iterator
     */
    Order *getOrders(CollationElementIterator &iter, int32_t &orderLength);
    UCollationResult compareUsingPartials(UCollator *coll, const char16_t source[], int32_t sLen, const char16_t target[], int32_t tLen, int32_t pieceSize, UErrorCode &status);

};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
