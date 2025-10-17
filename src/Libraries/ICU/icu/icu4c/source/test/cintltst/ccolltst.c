/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
 * Copyright (c) 1997-2012, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CCOLLTST.C
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda               Creation
*********************************************************************************
*/
#include <stdio.h>

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "cintltst.h"
#include "ccolltst.h"
#include "unicode/ucol.h"
#include "unicode/ustring.h"
#include "cmemory.h"

void addCollTest(TestNode** root);

void addCollTest(TestNode** root)
{
    addCollAPITest(root);
    addCurrencyCollTest(root);
#if !UCONFIG_NO_NORMALIZATION
    addNormTest(root);
#endif
    addGermanCollTest(root);
    addSpanishCollTest(root);
    addFrenchCollTest(root);
    addKannaCollTest(root);
    addTurkishCollTest(root);
    addEnglishCollTest(root);
    addFinnishCollTest(root);
    
    /* WEIVTODO: return tests here */
    addRuleBasedCollTest(root);
    addCollIterTest(root);
    addAllCollTest(root);
    addMiscCollTest(root);
#if !UCONFIG_NO_BREAK_ITERATION && !UCONFIG_NO_FILE_IO
    addSearchTest(root);
#endif
}



/*Internal functions used*/
static char* dumpSk(uint8_t *sourceKey, char *sk, size_t n) {
    uint32_t kLen = (uint32_t)strlen((const char *)sourceKey);
    uint32_t i = 0;
    
    *sk = 0;
    
    for(i = 0; i<kLen; i++) {
        snprintf(sk+2*i, n-2*i, "%02X", sourceKey[i]);
    }
    return sk;
}

static const char *getCompareResult(UCollationResult result)
{
    if (result == UCOL_LESS)
    {
        return "LESS";
    }
    else if (result == UCOL_EQUAL)
    {
        return "EQUAL";
    }
    else if (result == UCOL_GREATER)
    {
        return "GREATER";
    }
    return "invalid UCollationResult?";
}

void reportCResult( const UChar source[], const UChar target[], 
                         uint8_t *sourceKey, uint8_t *targetKey,
                         UCollationResult compareResult,
                         UCollationResult keyResult,
                         UCollationResult incResult,
                         UCollationResult expectedResult )
{
    if (expectedResult < -1 || expectedResult > 1)
    {
        log_err("***** invalid call to reportCResult ****\n");
        return;
    }

    if (compareResult != expectedResult)
    {
        log_err("Compare(%s , %s) returned: %s expected: %s\n", aescstrdup(source,-1), aescstrdup(target,-1),
            getCompareResult(compareResult), getCompareResult(expectedResult) );
    }

    if (incResult != expectedResult)
    {
        log_err("incCompare(%s , %s) returned: %s expected: %s\n", aescstrdup(source,-1), aescstrdup(target,-1),
            getCompareResult(incResult), getCompareResult(expectedResult) );
    }

    if (keyResult != expectedResult)
    {
        log_err("KeyCompare(%s , %s) returned: %s expected: %s\n", aescstrdup(source,-1), aescstrdup(target,-1), 
            getCompareResult(keyResult), getCompareResult(expectedResult) );
    }

    if (keyResult != compareResult)
    {
        log_err("difference between sortkey and compare result for (%s , %s) Keys: %s compare %s\n", aescstrdup(source,-1), aescstrdup(target,-1), 
            getCompareResult(keyResult), getCompareResult(compareResult));
    }

    if(keyResult != expectedResult || keyResult != compareResult)
    {
        char sk[10000];
        log_verbose("SortKey1: %s\n", dumpSk(sourceKey, sk, sizeof(sk)));
        log_verbose("SortKey2: %s\n", dumpSk(targetKey, sk, sizeof(sk)));
    }
}

#endif /* #if !UCONFIG_NO_COLLATION */
