/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
* File CDATTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda               Creation
*********************************************************************************
*/
/* C API TEST FOR DATE FORMAT */
#ifndef _CDATFRMTST
#define _CDATFRMTST

#include "unicode/utypes.h"
#include "unicode/udat.h"

#if !UCONFIG_NO_FORMATTING

#include "cintltst.h"

    /**
     * The functions used to test the Date format API
     **/
    static void TestDateFormat(void);
    static void TestRelativeDateFormat(void);

    /**
     * The function used to test API  udat_getSymbols(), udat_setSymbols() and udat_countSymbols()
     **/
    static void TestSymbols(void);

    /**
     * Test DateFormat(Calendar) API
     */
    static void TestDateFormatCalendar(void);

    /**
     * test subroutines used by TestSymbols
     **/
    static void VerifygetSymbols(UDateFormat*, UDateFormatSymbolType, int32_t, const char*);
    static void VerifysetSymbols(UDateFormat*, UDateFormatSymbolType, int32_t, const char*);
    static void VerifygetsetSymbols(UDateFormat*, UDateFormat*, UDateFormatSymbolType, int32_t);
    
    /**
     * test subroutine used by the testing functions
     **/
    static UChar* myNumformat(const UNumberFormat* numfor, double d);
    static int getCurrentYear(void);

    /**
     * Test DateFormat override number format API
     */
     static void TestOverrideNumberFormat(void);


#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
