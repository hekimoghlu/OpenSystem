/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
 * Copyright (c) 1997-2016, International Business Machines
 * Corporation and others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File CFORMTST.C
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda               Creation
*********************************************************************************
*/

/* FormatTest is a medium top level test for everything in the  C FORMAT API */

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"
#include "cformtst.h"

void addCalTest(TestNode**);
void addDateForTest(TestNode**);
void addDateTimePatternGeneratorTest(TestNode**);
void addDateIntervalFormatTest(TestNode**);
void addRelativeDateFormatTest(TestNode**);
void addNumForTest(TestNode**);
void addMsgForTest(TestNode**);
void addDateForRgrTest(TestNode**);
void addNumFrDepTest(TestNode**);
void addDtFrDepTest(TestNode**);
void addUtmsTest(TestNode**);
void addCurrencyTest(TestNode**);
void addPluralRulesTest(TestNode**);
void addURegionTest(TestNode** root);
void addUListFmtTest(TestNode** root);
void addUNumberFormatterTest(TestNode** root);
void addUFormattedValueTest(TestNode** root);
void addUNumberRangeFormatterTest(TestNode** root);
#if APPLE_ICU_CHANGES
// rdar://
void addMeasureFormatTest(TestNode** root);
void addCTZTest(TestNode** root);
#endif  // APPLE_ICU_CHANGES

void addFormatTest(TestNode** root);

void addFormatTest(TestNode** root)
{
    addCalTest(root);
    addDateForTest(root);
    addDateTimePatternGeneratorTest(root);
    addDateIntervalFormatTest(root);
#if !UCONFIG_NO_BREAK_ITERATION
    addRelativeDateFormatTest(root);
#endif /* !UCONFIG_NO_BREAK_ITERATION */
    addNumForTest(root);
    addNumFrDepTest(root);
    addMsgForTest(root);
    addDateForRgrTest(root);
    addDtFrDepTest(root);
    addUtmsTest(root);
    addCurrencyTest(root);
    addPluralRulesTest(root);
    addURegionTest(root);
    addUListFmtTest(root);
    addUNumberFormatterTest(root);
    addUFormattedValueTest(root);
    addUNumberRangeFormatterTest(root);
#if APPLE_ICU_CHANGES
// rdar://
    addMeasureFormatTest(root);
    addCTZTest(root);
#endif  // APPLE_ICU_CHANGES
}
/*Internal functions used*/

UChar* myDateFormat(UDateFormat* dat, UDate d1)
{
    UChar *result1=NULL;
    int32_t resultlength, resultlengthneeded;
    UErrorCode status = U_ZERO_ERROR;


    resultlength=0;
    resultlengthneeded=udat_format(dat, d1, NULL, resultlength, NULL, &status);
    if(status==U_BUFFER_OVERFLOW_ERROR)
    {
        status=U_ZERO_ERROR;
        resultlength=resultlengthneeded+1;
        result1=(UChar*)ctst_malloc(sizeof(UChar) * resultlength);
        udat_format(dat, d1, result1, resultlength, NULL, &status);
    }
    if(U_FAILURE(status))
    {
        log_err("Error in formatting using udat_format(.....): %s\n", myErrorName(status));
        return 0;
    }
    return result1;

}

#endif /* #if !UCONFIG_NO_FORMATTING */
