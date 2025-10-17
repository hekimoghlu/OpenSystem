/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
 * Copyright (c) 1997-2014, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CNUMTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda              Creation
*********************************************************************************
*/
/* C API TEST FOR NUMBER FORMAT */
#ifndef _CNUMFRMTST
#define _CNUMFRMTST

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"


/**
 * The function used to test the Number format API
 **/
static void TestNumberFormat(void);

/**
 * The function used to test parsing of numbers in UNUM_SPELLOUT style
 **/
static void TestSpelloutNumberParse(void);

/**
 * The function used to test significant digits in the Number format API
 **/
static void TestSignificantDigits(void);

/**
 * The function used to test Number format API rounding with significant digits
 **/
static void TestSigDigRounding(void);

/**
 * The function used to test the Number format API with padding
 **/
static void TestNumberFormatPadding(void);

/**
 * The function used to test the Number format API with padding
 **/
static void TestInt64Format(void);

static void TestNonExistentCurrency(void);

/**
 * Test RBNF access through unumfmt APIs.
 **/
static void TestRBNFFormat(void);

/**
 * Test some Currency stuff
 **/
static void TestCurrencyRegression(void);

/**
 * Test strict parsing of "0"
 **/
static void TestParseZero(void);

/**
 * Test cloning formatter with RBNF
 **/
static void TestCloneWithRBNF(void);

/**
 * Test the Currency Usage Implementations
 **/
static void TestCurrencyUsage(void);
#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
