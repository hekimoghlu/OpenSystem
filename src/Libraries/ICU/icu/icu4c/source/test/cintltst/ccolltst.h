/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
 * Copyright (c) 1997-2005, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CCOLLTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda               Creation
*********************************************************************************
*/
#ifndef _CCOLLTST
#define _CCOLLTST

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "cintltst.h"
#include "unicode/ucol.h"

/* Internal Functions used*/

void reportCResult( const UChar source[], const UChar target[], 
                         uint8_t *sourceKey, uint8_t *targetKey,
                         UCollationResult compareResult,
                         UCollationResult keyResult,
                         UCollationResult incResult,
                         UCollationResult expectedResult );

UChar* appendCompareResult(UCollationResult result, UChar* target);

void addCollAPITest(TestNode**);
void addCurrencyCollTest(TestNode**);
void addNormTest(TestNode**);
void addDanishCollTest(TestNode**);
void addGermanCollTest(TestNode**);
void addSpanishCollTest(TestNode**);
void addFrenchCollTest(TestNode**);
void addKannaCollTest(TestNode**);
void addTurkishCollTest(TestNode**);
void addEnglishCollTest(TestNode**);
void addFinnishCollTest(TestNode**);

void addRuleBasedCollTest(TestNode**);
void addCollIterTest(TestNode**);
void addAllCollTest(TestNode**);
void addMiscCollTest(TestNode**);
void addSearchTest(TestNode**);

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
