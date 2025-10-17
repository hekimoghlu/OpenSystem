/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
*******************************************************************************
*
*   Copyright (C) 1998-2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*
* File error.c
*
* Modification History:
*
*   Date        Name        Description
*   05/28/99    stephen     Creation.
*******************************************************************************
*/

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include "cstring.h"
#include "errmsg.h"
#include "toolutil.h"

U_CFUNC void error(uint32_t linenumber, const char *msg, ...)
{
    va_list va;

    va_start(va, msg);
    fprintf(stderr, "%s:%u: ", gCurrentFileName, (int)linenumber);
    vfprintf(stderr, msg, va);
    fprintf(stderr, "\n");
    va_end(va);
}

static UBool gShowWarning = true;

U_CFUNC void setShowWarning(UBool val)
{
    gShowWarning = val;
}

U_CFUNC UBool getShowWarning(void){
    return gShowWarning;
}

static UBool gStrict =false;
U_CFUNC UBool isStrict(void){
    return gStrict;
}
U_CFUNC void setStrict(UBool val){
    gStrict = val;
}
static UBool gVerbose =false;
U_CFUNC UBool isVerbose(void){
    return gVerbose;
}
U_CFUNC void setVerbose(UBool val){
    gVerbose = val;
}
U_CFUNC void warning(uint32_t linenumber, const char *msg, ...)
{
    if (gShowWarning)
    {
        va_list va;

        va_start(va, msg);
        fprintf(stderr, "%s:%u: warning: ", gCurrentFileName, (int)linenumber);
        vfprintf(stderr, msg, va);
        fprintf(stderr, "\n");
        va_end(va);
    }
}
