/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
* Copyright (C) 2011-2012, International Business Machines Corporation and    *
* others. All Rights Reserved.                                                *
*******************************************************************************
*/
#ifndef __TZGNAMES_H
#define __TZGNAMES_H

/**
 * \file 
 * \brief C API: Time zone generic names classes
 */

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/locid.h"
#include "unicode/unistr.h"
#include "unicode/tzfmt.h"
#include "unicode/tznames.h"

U_CDECL_BEGIN

typedef enum UTimeZoneGenericNameType {
    UTZGNM_UNKNOWN      = 0x00,
    UTZGNM_LOCATION     = 0x01,
    UTZGNM_LONG         = 0x02,
    UTZGNM_SHORT        = 0x04
} UTimeZoneGenericNameType;

U_CDECL_END

U_NAMESPACE_BEGIN

class TimeZone;
struct TZGNCoreRef;

class U_I18N_API TimeZoneGenericNames : public UMemory {
public:
    virtual ~TimeZoneGenericNames();

    static TimeZoneGenericNames* createInstance(const Locale& locale, UErrorCode& status);

    virtual bool operator==(const TimeZoneGenericNames& other) const;
    virtual bool operator!=(const TimeZoneGenericNames& other) const {return !operator==(other);}
    virtual TimeZoneGenericNames* clone() const;

    UnicodeString& getDisplayName(const TimeZone& tz, UTimeZoneGenericNameType type,
                        UDate date, UnicodeString& name) const;

    UnicodeString& getGenericLocationName(const UnicodeString& tzCanonicalID, UnicodeString& name) const;

    int32_t findBestMatch(const UnicodeString& text, int32_t start, uint32_t types,
        UnicodeString& tzID, UTimeZoneFormatTimeType& timeType, UErrorCode& status) const;

private:
    TimeZoneGenericNames();
    TZGNCoreRef* fRef;
};

U_NAMESPACE_END
#endif
#endif
