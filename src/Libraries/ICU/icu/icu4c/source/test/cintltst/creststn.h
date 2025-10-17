/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
 * Copyright (c) 1997-2013, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/***************************************************************************
*
* File CRESTST.H
*
* Modification History:
*        Name               Date               Description
*   Madhu Katragadda    05/09/2000   Ported Tests for New ResourceBundle API
*   Madhu Katragadda    05/24/2000   Added new tests to test RES_BINARY for collationElements
*************************************************************************************************
*/
#ifndef _CRESTSTN
#define _CRESTSTN

#include "unicode/utypes.h"
#include "unicode/ures.h"

/* C TEST FOR NEW RESOURCEBUNDLE API*/
#include "cintltst.h"

/*
 * Test wrapper for ures_getStringXYZ(), for testing other variants of
 * these functions as well.
 * If index>=0, calls ures_getStringByIndex().
 * If key!=NULL, calls ures_getStringByKey().
 */
extern const UChar *
tres_getString(const UResourceBundle *resB,
               int32_t index, const char *key,
               int32_t *length,
               UErrorCode *status);

void addNEWResourceBundleTest(TestNode**);

/**
*Perform several extensive tests using the subtest routine testTag
*/
static void TestResourceBundles(void);
/** 
* Test construction of ResourceBundle accessing a custom test resource-file
**/
static void TestConstruction1(void);

static void TestAliasConflict(void);

static void TestFallback(void);

static void TestPreventFallback(void);

static void TestBinaryCollationData(void);

static void TestNewTypes(void);

static void TestEmptyTypes(void);

static void TestAPI(void);

static void TestErrorConditions(void);

static void TestGetVersion(void);

static void TestGetVersionColl(void);

static void TestEmptyBundle(void);

static void TestDirectAccess(void);

static void TestTicket9804(void);

static void TestResourceLevelAliasing(void);

static void TestErrorCodes(void);

static void TestJB3763(void);

static void TestXPath(void);

/**
* extensive subtests called by TestResourceBundles
**/
static UBool testTag(const char* frag, UBool in_Root, UBool in_te, UBool in_te_IN);

static void record_pass(void);
static void record_fail(void);


#endif
