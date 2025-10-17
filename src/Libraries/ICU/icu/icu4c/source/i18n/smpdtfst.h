/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
* Copyright (C) 2009-2013, International Business Machines Corporation and    *
* others. All Rights Reserved.                                                *
*******************************************************************************
*
* This file contains declarations for the class SimpleDateFormatStaticSets
*
* SimpleDateFormatStaticSets holds the UnicodeSets that are needed for lenient
* parsing of literal characters in date/time strings.
********************************************************************************
*/

#ifndef SMPDTFST_H
#define SMPDTFST_H

#include "unicode/uobject.h"
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/udat.h"

U_NAMESPACE_BEGIN

class  UnicodeSet;


class SimpleDateFormatStaticSets : public UMemory
{
public:
    SimpleDateFormatStaticSets(UErrorCode &status);
    ~SimpleDateFormatStaticSets();
    
    static void    initSets(UErrorCode *status);
    static UBool   cleanup();
    
    static UnicodeSet *getIgnorables(UDateFormatField fieldIndex);
    
private:
    UnicodeSet *fDateIgnorables;
    UnicodeSet *fTimeIgnorables;
    UnicodeSet *fOtherIgnorables;
};


U_NAMESPACE_END

#endif   // #if !UCONFIG_NO_FORMATTING
#endif   // SMPDTFST_H
