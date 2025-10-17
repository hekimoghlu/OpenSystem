/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

// Â© 2022 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "iso8601cal.h"
#include "unicode/gregocal.h"

U_NAMESPACE_BEGIN

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(ISO8601Calendar)

ISO8601Calendar::ISO8601Calendar(const Locale& aLocale, UErrorCode& success)
:   GregorianCalendar(aLocale, success)
{
    UErrorCode tempStatus = U_ZERO_ERROR;
    int32_t length = aLocale.getKeywordValue("fw", nullptr, 0, tempStatus) +
        aLocale.getKeywordValue("rg", nullptr, 0, tempStatus);
    // Do not set first day of week for iso8601 to Monday if we have fw or rg keywords
    // and let the value set by the Calendar constructor to take care of it.
    if (U_SUCCESS(tempStatus) && length == 0) {
        setFirstDayOfWeek(UCAL_MONDAY);
    }
    setMinimalDaysInFirstWeek(4);
}

ISO8601Calendar::~ISO8601Calendar()
{
}

ISO8601Calendar* ISO8601Calendar::clone() const
{
    return new ISO8601Calendar(*this);
}

const char *ISO8601Calendar::getType() const
{
    return "iso8601";
}

U_NAMESPACE_END

#endif
