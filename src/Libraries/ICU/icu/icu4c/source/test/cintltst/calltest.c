/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
 * Copyright (c) 1996-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CALLTEST.C
*
* Modification History:
*   Creation:   Madhu Katragadda 
*********************************************************************************
*/
/* THE FILE WHERE ALL C API TESTS ARE ADDED TO THE ROOT */


#include "cintltst.h"

void addUtility(TestNode** root);
void addBreakIter(TestNode** root);
void addStandardNamesTest(TestNode **root);
void addFormatTest(TestNode** root);
void addConvert(TestNode** root);
void addCollTest(TestNode** root);
void addComplexTest(TestNode** root);
void addBidiTransformTest(TestNode** root);
void addUDataTest(TestNode** root);
void addUTF16Test(TestNode** root);
void addUTF8Test(TestNode** root);
void addUTransTest(TestNode** root);
void addPUtilTest(TestNode** root);
void addUCharTransformTest(TestNode** root);
void addUSetTest(TestNode** root);
void addUStringPrepTest(TestNode** root);
void addIDNATest(TestNode** root);
void addHeapMutexTest(TestNode **root);
void addUTraceTest(TestNode** root);
void addURegexTest(TestNode** root);
void addUTextTest(TestNode** root);
void addUCsdetTest(TestNode** root);
void addCnvSelTest(TestNode** root);
void addUSpoofTest(TestNode** root);
#if !UCONFIG_NO_FORMATTING
void addGendInfoForTest(TestNode** root);
#endif
#if APPLE_ICU_CHANGES
// rdar://
void addDataDumper(TestNode** root); // Apple addition
#endif  // APPLE_ICU_CHANGES

void addAllTests(TestNode** root)
{
    addCnvSelTest(root);
    addUDataTest(root);
    addHeapMutexTest(root);
    addUTF16Test(root);
    addUTF8Test(root);
    addUtility(root);
    addUTraceTest(root);
    addUTextTest(root);
    addConvert(root);
    addUCharTransformTest(root);
    addStandardNamesTest(root);
    addUCsdetTest(root);
    addComplexTest(root);
    addBidiTransformTest(root);
    addUSetTest(root);
#if !UCONFIG_NO_IDNA
    addUStringPrepTest(root);
    addIDNATest(root);
#endif
#if !UCONFIG_NO_REGULAR_EXPRESSIONS
    addURegexTest(root);
#endif
#if !UCONFIG_NO_BREAK_ITERATION
    addBreakIter(root);
#endif
#if !UCONFIG_NO_FORMATTING
    addFormatTest(root);
#endif
#if !UCONFIG_NO_COLLATION
    addCollTest(root);
#endif
#if !UCONFIG_NO_TRANSLITERATION
    addUTransTest(root);
#endif
#if !UCONFIG_NO_REGULAR_EXPRESSIONS && !UCONFIG_NO_NORMALIZATION
    addUSpoofTest(root);
#endif
    addPUtilTest(root);
#if !UCONFIG_NO_FORMATTING
    addGendInfoForTest(root);
#endif
#if APPLE_ICU_CHANGES
// rdar://
    addDataDumper(root);
#endif  // APPLE_ICU_CHANGES
}
