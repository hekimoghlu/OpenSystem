/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
*   Copyright (C) 2003-2006, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:   regexc.cpp
*   description: The purpose of this function is to separate the codepage
*       conversion from the rest of the uregex_ API. This can removes any
*       dependency on codepage conversion, which reduces the overhead of 
*/

#include "unicode/uregex.h"
#include "unicode/unistr.h"

U_NAMESPACE_USE

//----------------------------------------------------------------------------------------
//
//    uregex_openC
//
//----------------------------------------------------------------------------------------
#if !UCONFIG_NO_CONVERSION && !UCONFIG_NO_REGULAR_EXPRESSIONS

U_CAPI URegularExpression * U_EXPORT2
uregex_openC( const char           *pattern,
                    uint32_t        flags,
                    UParseError    *pe,
                    UErrorCode     *status) {
    if (U_FAILURE(*status)) {
        return nullptr;
    }
    if (pattern == nullptr) {
        *status = U_ILLEGAL_ARGUMENT_ERROR;
        return nullptr;
    }

    UnicodeString patString(pattern);
    return uregex_open(patString.getBuffer(), patString.length(), flags, pe, status);
}
#endif
