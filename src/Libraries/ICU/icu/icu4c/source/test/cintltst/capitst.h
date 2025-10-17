/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
 * Copyright (c) 1997-2013 International Business Machines 
 * Corporation and others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CAPITEST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Converted to C
*     Brian Rower                 Added TestOpenVsOpenRules
*********************************************************************************
*//* C API TEST For COLLATOR */

#ifndef _CCOLLAPITST
#define _CCOLLAPITST

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "cintltst.h"
#include "callcoll.h"
#define MAX_TOKEN_LEN 16


    /**
     * error reporting utility method
     **/

    static void doAssert(int condition, const char *message);
    /**
     * Collator Class Properties
     * ctor, dtor, createInstance, compare, getStrength/setStrength
     * getDecomposition/setDecomposition, getDisplayName
     */
    void TestProperty(void);
    /**
     * Test RuleBasedCollator and getRules
     **/
    void TestRuleBasedColl(void);
    
    /**
     * Test compare
     **/
    void TestCompare(void);
    /**
     * Test hashCode functionality
     **/
    void TestHashCode(void);
    /**
     * Tests the constructor and numerous other methods for CollationKey
     **/
   void TestSortKey(void);
    /**
     * test the CollationElementIterator methods
     **/
   void TestElemIter(void);
    /**
     * Test ucol_getAvailable and ucol_countAvailable()
     **/
    void TestGetAll(void);
    /**
     * Test ucol_GetDefaultRules ()
    void TestGetDefaultRules(void);
     **/

    void TestDecomposition(void);
    /**
     * Test ucol_safeClone ()
     **/    
    void TestSafeClone(void);

    /**
     * Test ucol_clone ()
     **/
    void TestClone(void);

    /**
     * Test ucol_cloneBinary(), ucol_openBinary()
     **/
    void TestCloneBinary(void);

    /**
     * Test ucol_open() vs. ucol_openRules()
     **/
    void TestOpenVsOpenRules(void);

    /**
     * Test getting bounds for a sortkey
     */
    void TestBounds(void);

    /**
     * Test ucol_getLocale function
     */
    void TestGetLocale(void);

    /**
     * Test buffer overrun while having smaller buffer for sortkey (j1865)
     */
    void TestSortKeyBufferOverrun(void);
    /**
     * Test getting and setting of attributes
     */
    void TestGetSetAttr(void);
    /**
     * Test getTailoredSet
     */
    void TestGetTailoredSet(void);

    /**
     * Test mergeSortKeys
     */
    void TestMergeSortKeys(void);

    /** 
     * test short string and collator identifier functions
     */
    static void TestShortString(void);

    /** 
     * test getContractions and getUnsafeSet
     */
    static void TestGetContractionsAndUnsafes(void);

    /**
     * Test funny stuff with open binary
     */
    static void TestOpenBinary(void);

    /**
     * Test getKeywordValuesForLocale API
     */
    static void TestGetKeywordValuesForLocale(void);

    /**
     * test strcoll with null arg
     */
    static void TestStrcollNull(void);
 
    /**
     * Simple test for ICU-21460.  The issue affects all components, but was originally reported against collation.
     */
    static void TestLocaleIDWithUnderscoreAndExtension(void);

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
