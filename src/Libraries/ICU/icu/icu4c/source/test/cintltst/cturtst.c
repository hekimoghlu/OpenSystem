/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
 * Copyright (c) 1997-2009, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CTURTST.C
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda            Ported for C API
*********************************************************************************/
/**
 * CollationTurkishTest is a third level test class.  This tests the locale
 * specific primary, secondary and tertiary rules.  For example, the ignorable
 * character '-' in string "black-bird".  The en_US locale uses the default
 * collation rules as its sorting sequence.
 */

#include <stdlib.h>

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "unicode/ucol.h"
#include "unicode/uloc.h"
#include "cintltst.h"
#include "ccolltst.h"
#include "callcoll.h"
#include "cturtst.h"
#include "unicode/ustring.h"
#include "string.h"

static UCollator *myCollation;
const static UChar testSourceCases[][MAX_TOKEN_LEN] = {
    {0x0073/*'s'*/, 0x0327, 0x0000},
    {0x0076/*'v'*/, 0x00E4, 0x0074/*'t'*/, 0x0000},
    {0x006f/*'o'*/, 0x006c/*'l'*/, 0x0064/*'d'*/, 0x0000},
    {0x00FC, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064/*'d'*/, 0x0000},
    {0x0068/*'h'*/, 0x011E, 0x0061/*'a'*/, 0x006c/*'l'*/, 0x0074/*'t'*/, 0x0000},
    {0x0073/*'s'*/, 0x0074/*'t'*/, 0x0072/*'r'*/, 0x0065/*'e'*/, 0x0073/*'s'*/, 0x015E, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0131, 0x0064/*'d'*/, 0x0000},
    {0x0069/*'i'*/, 0x0064/*'d'*/, 0x0065/*'e'*/, 0x0061/*'a'*/, 0x0000},
    {0x00FC, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064 /*d'*/, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0131, 0x0064 /*d'*/, 0x0000},
    {0x0069/*'i'*/, 0x0064/*'d'*/, 0x0065/*'e'*/, 0x0061/*'a'*/, 0x0000},
};

const static UChar testTargetCases[][MAX_TOKEN_LEN] = {
    {0x0075/*'u'*/, 0x0308, 0x0000},
    {0x0076/*'v'*/, 0x0062/*'b'*/, 0x0074/*'t'*/, 0x0000},
    {0x00D6, 0x0061/*'a'*/, 0x0079/*'y'*/, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064 /*d'*/, 0x0000},
    {0x0068/*'h'*/, 0x0061/*'a'*/,  0x006c/*'l'*/, 0x0074/*'t'*/, 0x0000},
    {0x015E, 0x0074/*'t'*/, 0x0072/*'r'*/, 0x0065/*'e'*/, 0x015E, 0x0073/*'s'*/, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064 /*d'*/, 0x0000},
    {0x0049/*'I'*/, 0x0064/*'d'*/, 0x0065/*'e'*/, 0x0061/*'a'*/, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064 /*d'*/, 0x0000},
    {0x0076/*'v'*/, 0x006f/*'o'*/, 0x0069/*'i'*/, 0x0064 /*d'*/, 0x0000},
    {0x0049/*'I'*/, 0x0064/*'d'*/, 0x0065/*'e'*/, 0x0061/*'a'*/, 0x0000},
};

const static UCollationResult results[] = {
    UCOL_LESS,
    UCOL_LESS,
    UCOL_LESS,
    UCOL_LESS,
    UCOL_GREATER,
    UCOL_LESS,
    UCOL_LESS,
    UCOL_GREATER,
    /* test priamry > 8 */
    UCOL_LESS,
    UCOL_LESS, /*Turkish translator made a primary difference between dotted and dotless I */
    UCOL_GREATER
};



void addTurkishCollTest(TestNode** root)
{
    
    addTest(root, &TestPrimary, "tscoll/cturtst/TestPrimary");
    addTest(root, &TestTertiary, "tscoll/cturtst/TestTertiary");


}

static void TestTertiary(void)
{
    
    int32_t i;

    UErrorCode status = U_ZERO_ERROR;
    myCollation = ucol_open("tr", &status);
    if(U_FAILURE(status)){
        log_err_status(status, "ERROR: in creation of rule based collator: %s\n", myErrorName(status));
        return;
    }
    log_verbose("Testing Turkish Collation with Tertiary strength\n");
    ucol_setStrength(myCollation, UCOL_TERTIARY);
    for (i = 0; i < 8 ; i++)
    {
        doTest(myCollation, testSourceCases[i], testTargetCases[i], results[i]);
    }
    ucol_close(myCollation);
}

static void TestPrimary(void)
{
    
    int32_t i;

    UErrorCode status = U_ZERO_ERROR;
    myCollation = ucol_open("tr", &status);
    if(U_FAILURE(status)){
        log_err_status(status, "ERROR: in creation of rule based collator: %s\n", myErrorName(status));
        return;
    }
    log_verbose("Testing Turkish Collation with Primary strength\n");
    ucol_setStrength(myCollation, UCOL_PRIMARY);
    for (i = 8; i < 11; i++)
    {
        doTest(myCollation, testSourceCases[i], testTargetCases[i], results[i]);
    }
    ucol_close(myCollation);
}

#endif /* #if !UCONFIG_NO_COLLATION */
