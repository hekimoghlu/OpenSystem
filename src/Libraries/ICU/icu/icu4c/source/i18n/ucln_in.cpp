/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
******************************************************************************
*                                                                            *
* Copyright (C) 2001-2014, International Business Machines                   *
*                Corporation and others. All Rights Reserved.                *
*                                                                            *
******************************************************************************
*   file name:  ucln_in.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2001July05
*   created by: George Rhoten
*/

#include "ucln.h"
#include "ucln_in.h"
#include "mutex.h"
#include "uassert.h"

/**  Auto-client for UCLN_I18N **/
#define UCLN_TYPE UCLN_I18N
#include "ucln_imp.h"

/* Leave this copyright notice here! It needs to go somewhere in this library. */
static const char copyright[] = U_COPYRIGHT_STRING;

static cleanupFunc *gCleanupFunctions[UCLN_I18N_COUNT];

static UBool U_CALLCONV i18n_cleanup()
{
    int32_t libType = UCLN_I18N_START;
    (void)copyright;   /* Suppress unused variable warning with clang. */

    while (++libType<UCLN_I18N_COUNT) {
        if (gCleanupFunctions[libType])
        {
            gCleanupFunctions[libType]();
            gCleanupFunctions[libType] = nullptr;
        }
    }
#if !UCLN_NO_AUTO_CLEANUP && (defined(UCLN_AUTO_ATEXIT) || defined(UCLN_AUTO_LOCAL))
    ucln_unRegisterAutomaticCleanup();
#endif
    return true;
}

void ucln_i18n_registerCleanup(ECleanupI18NType type,
                               cleanupFunc *func) {
    U_ASSERT(UCLN_I18N_START < type && type < UCLN_I18N_COUNT);
    {
        icu::Mutex m;   // See ticket 10295 for discussion.
        ucln_registerCleanup(UCLN_I18N, i18n_cleanup);
        if (UCLN_I18N_START < type && type < UCLN_I18N_COUNT) {
            gCleanupFunctions[type] = func;
        }
    }
#if !UCLN_NO_AUTO_CLEANUP && (defined(UCLN_AUTO_ATEXIT) || defined(UCLN_AUTO_LOCAL))
    ucln_registerAutomaticCleanup();
#endif
}

