/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
*   Copyright (C) 1997-2006, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
*******************************************************************************
*   file name:  resbund_cnv.cpp
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2004aug25
*   created by: Markus W. Scherer
*
*   Character conversion functions moved here from resbund.cpp
*/

#include "unicode/utypes.h"
#include "unicode/resbund.h"
#include "uinvchar.h"

U_NAMESPACE_BEGIN

ResourceBundle::ResourceBundle( const UnicodeString&    path,
                                const Locale&           locale,
                                UErrorCode&              error)
                                :UObject(), fLocale(nullptr)
{
    constructForLocale(path, locale, error);
}

ResourceBundle::ResourceBundle( const UnicodeString&    path,
                                UErrorCode&              error)
                                :UObject(), fLocale(nullptr)
{
    constructForLocale(path, Locale::getDefault(), error);
}

void 
ResourceBundle::constructForLocale(const UnicodeString& path,
                                   const Locale& locale,
                                   UErrorCode& error)
{
    if (path.isEmpty()) {
        fResource = ures_open(nullptr, locale.getName(), &error);
    }
    else {
        UnicodeString nullTerminatedPath(path);
        nullTerminatedPath.append(static_cast<char16_t>(0));
        fResource = ures_openU(nullTerminatedPath.getBuffer(), locale.getName(), &error);
    }
}

U_NAMESPACE_END
