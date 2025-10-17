/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
/********************************************************************************
*
* File CUTILTST.C
*
* Modification History:
*        Name                     Description
*     Madhu Katragadda               Creation
*********************************************************************************
*/
#include "cintltst.h"

void addLocaleTest(TestNode**);
void addULocaleTest(TestNode**);
void addLocaleBuilderTest(TestNode**);
void addCLDRTest(TestNode**);
void addUnicodeTest(TestNode**);
void addUStringTest(TestNode**);
void addCaseTest(TestNode**);
void addResourceBundleTest(TestNode**);
void addNEWResourceBundleTest(TestNode**);
void addHashtableTest(TestNode** root);
void addCStringTest(TestNode** root);
void addTrieTest(TestNode** root);
void addTrie2Test(TestNode** root);
void addUCPTrieTest(TestNode** root);
void addEnumerationTest(TestNode** root);
void addPosixTest(TestNode** root);
void addSortTest(TestNode** root);

void addUtility(TestNode** root);

void addUtility(TestNode** root)
{
    addCStringTest(root);
    addTrieTest(root);
    addTrie2Test(root);
    addUCPTrieTest(root);
    addLocaleTest(root);
    addLocaleBuilderTest(root);
    addULocaleTest(root);
    addCLDRTest(root);
    addUnicodeTest(root);
    addUStringTest(root);
    addCaseTest(root);
    addResourceBundleTest(root);
    addNEWResourceBundleTest(root);
    addHashtableTest(root);
    addEnumerationTest(root);
    addPosixTest(root);
    addSortTest(root);
}
