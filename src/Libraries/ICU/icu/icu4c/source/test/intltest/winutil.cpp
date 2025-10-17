/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
/*
********************************************************************************
*   Copyright (C) 2005-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
********************************************************************************
*
* File WINUTIL.CPP
*
********************************************************************************
*/

#include "unicode/utypes.h"

#if U_PLATFORM_HAS_WIN32_API

#if !UCONFIG_NO_FORMATTING

#include "cmemory.h"
#include "winutil.h"
#include "locmap.h"
#include "unicode/uloc.h"

#   define WIN32_LEAN_AND_MEAN
#   define VC_EXTRALEAN
#   define NOUSER
#   define NOSERVICE
#   define NOIME
#   define NOMCX
#   include <windows.h>
#   include <stdio.h>
#   include <string.h>

static Win32Utilities::LCIDRecord *lcidRecords = nullptr;
static int32_t lcidCount  = 0;
static int32_t lcidMax = 0;

// TODO: Note that this test will skip locale names and only hit locales with assigned LCIDs
BOOL CALLBACK EnumLocalesProc(LPSTR lpLocaleString)
{
    char localeID[ULOC_FULLNAME_CAPACITY];
	int32_t localeIDLen;
    UErrorCode status = U_ZERO_ERROR;

    if (lcidCount >= lcidMax) {
        Win32Utilities::LCIDRecord *newRecords = new Win32Utilities::LCIDRecord[lcidMax + 32];

        for (int i = 0; i < lcidMax; i += 1) {
            newRecords[i] = lcidRecords[i];
        }

        delete[] lcidRecords;
        lcidRecords = newRecords;
        lcidMax += 32;
    }

    sscanf(lpLocaleString, "%8x", &lcidRecords[lcidCount].lcid);

    localeIDLen = uprv_convertToPosix(lcidRecords[lcidCount].lcid, localeID, UPRV_LENGTHOF(localeID), &status);
    if (U_SUCCESS(status)) {
        lcidRecords[lcidCount].localeID = new char[localeIDLen + 1];
        memcpy(lcidRecords[lcidCount].localeID, localeID, localeIDLen);
        lcidRecords[lcidCount].localeID[localeIDLen] = 0;
    } else {
        lcidRecords[lcidCount].localeID = nullptr;
    }

    lcidCount += 1;

    return true;
}

// TODO: Note that this test will skip locale names and only hit locales with assigned LCIDs
Win32Utilities::LCIDRecord *Win32Utilities::getLocales(int32_t &localeCount)
{
    LCIDRecord *result;

    EnumSystemLocalesA(EnumLocalesProc, LCID_INSTALLED);

    localeCount = lcidCount;
    result      = lcidRecords;

    lcidCount = lcidMax = 0;
    lcidRecords = nullptr;

    return result;
}

void Win32Utilities::freeLocales(LCIDRecord *records)
{
    for (int i = 0; i < lcidCount; i++) {
        delete lcidRecords[i].localeID;
    }
    delete[] records;
}

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif /* U_PLATFORM_HAS_WIN32_API */
