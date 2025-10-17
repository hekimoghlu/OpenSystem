/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
* File CALLCOLL.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda              Ported to C
*********************************************************************************
*/
/**
 * CollationDummyTest is a third level test class.  This tests creation of 
 * a customized collator object.  For example, number 1 to be sorted 
 * equlivalent to word 'one'.
 */
#ifndef _CALLCOLLTST
#define _CALLCOLLTST

#include "unicode/utypes.h"
#include "unicode/ucoleitr.h"

#if !UCONFIG_NO_COLLATION

#include "cintltst.h"

#define RULE_BUFFER_LEN 8192

struct OrderAndOffset
{
    int32_t order;
    int32_t offset;
};

typedef struct OrderAndOffset OrderAndOffset;

    /* tests comparison of custom collation with different strengths */
void doTest(UCollator*, const UChar* source, const UChar* target, UCollationResult result);
/* verify that iterating forward and backwards over the string yields same CEs */
void backAndForth(UCollationElements *iter);
/* gets an array of CEs for a string in UCollationElements iterator. */
OrderAndOffset* getOrders(UCollationElements *iter, int32_t *orderLength);

void genericOrderingTestWithResult(UCollator *coll, const char * const s[], uint32_t size, UCollationResult result);
void genericOrderingTest(UCollator *coll, const char * const s[], uint32_t size);
void genericLocaleStarter(const char *locale, const char * const s[], uint32_t size);
void genericLocaleStarterWithResult(const char *locale, const char * const s[], uint32_t size, UCollationResult result);
void genericLocaleStarterWithOptions(const char *locale, const char * const s[], uint32_t size, const UColAttribute *attrs, const UColAttributeValue *values, uint32_t attsize);
void genericLocaleStarterWithOptionsAndResult(const char *locale, const char * const s[], uint32_t size, const UColAttribute *attrs, const UColAttributeValue *values, uint32_t attsize, UCollationResult result);
void genericRulesStarterWithResult(const char *rules, const char * const s[], uint32_t size, UCollationResult result);
void genericRulesStarter(const char *rules, const char * const s[], uint32_t size);
void genericRulesStarterWithOptionsAndResult(const char *rules, const char * const s[], uint32_t size, const UColAttribute *attrs, const UColAttributeValue *values, uint32_t attsize, UCollationResult result);
UBool hasCollationElements(const char *locName);


#endif /* #if !UCONFIG_NO_COLLATION */

#endif
