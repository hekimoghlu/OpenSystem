/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
 * Copyright (C) 1996-2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 *******************************************************************************
 */

#ifndef UATIMEZONE_H
#define UATIMEZONE_H

#include "unicode/utypes.h"

#if APPLE_ICU_CHANGES
// rdar://125359625 (Provide access to ICU's C++ TimeZone API via C interface)


#include "unicode/ucal.h"

typedef void* UTimeZone;

U_CAPI UTimeZone* U_EXPORT2
uatimezone_open(const UChar*   zoneID,
                int32_t        len,
                UErrorCode*    status);


U_CAPI void U_EXPORT2
uatimezone_close(UTimeZone* tz);

enum UTimeZoneDisplayNameType {
    UTIMEZONE_SHORT,
    UTIMEZONE_LONG,
    UTIMEZONE_SHORT_DST,
    UTIMEZONE_LONG_DST,
};

typedef enum UTimeZoneDisplayNameType UTimeZoneDisplayNameType;

/**
 * C wrapper for TimeZone::getDisplayName().  Returns a name of this time zone
 * suitable for presentation to the user in the specified locale.
 * @param tz The time zone whose display name we want.
 * @param type The type of display name to return: short or long, in DST or not in DST.
 * @param locale The locale for the requested display name.
 * @param result A buffer where the display name is to be written.
 * @param resultLength The capacity (in UChars) of the result buffer.
 * @param status The error code.
 */
U_CAPI int32_t U_EXPORT2
uatimezone_getDisplayName(const UTimeZone*          tz,
                          UTimeZoneDisplayNameType  type,
                          const char                *locale,
                          UChar*                    result,
                          int32_t                   resultLength,
                          UErrorCode*               status);


/**
 * Equivalent to `ucal_getTimeZoneTransitionDate()`, but calls through to
 * `TimeZone` for performance.
 * @param tz The UTimeZone to query.
 * @param date A base date.  The function returns the nearest transition date to this date,
 * in the direction specified by `type`.  The result can be a transition either into or out of DST.
 * @param type The direction of the desired transition.
 * @param transition A pointer to a UDate to be set to the transition time.  If the function
 * returns false, the value set is unspecified.
 * @param status A pointer to a UErrorCode to receive any errors.
 * @return True if a value transition time is set in `*transition`, false otherwise.
 */
U_CAPI UBool U_EXPORT2
uatimezone_getTimeZoneTransitionDate(const UTimeZone*        tz,
                                     UDate                   date,
                                     UTimeZoneTransitionType type,
                                     UDate*                  transition,
                                     UErrorCode*             status);

/**
 * C wrapper for TimeZone::getOffset().  Returns the time zone raw and GMT offset for the
 * given moment in time.
 * @param tz The time zone to use for calculating the offsets.
 * @param date The moment in time for which to return offsets, in either GMT or local wall time.
 * @param local If true, `date` is in local wall time; otherwise, it is in GMT.
 * @param rawOffset Pointer to a variable that will receive the raw ofset-- that is, the
 * time zone offset with no DST adjustments.
 * @param dstOffset Pointer to a variable that will receive the DST offset-- that is,
 * the offset to be added to the `rawOffset` to obtain the total offset between local and GMT time.
 * If DST is not in effect, this value is zero; otherwise, it is a positive value, typically one hour.
 * @param status The error code.
 */
U_CAPI void U_EXPORT2
uatimezone_getOffset(const UTimeZone*     tz,
                     UDate                date,
                     UBool                local,
                     int32_t*             rawOffset,
                     int32_t*             dstOffset,
                     UErrorCode*          status);

/**
 * C wrapper for BasicTimeZone::getOffsetFromLocal().
 * (Which has no actual documentation...)
 */
U_CAPI void U_EXPORT2 uatimezone_getOffsetFromLocal(const UTimeZone      *tz,
                                                    UTimeZoneLocalOption nonExistingTimeOpt,
                                                    UTimeZoneLocalOption duplicatedTimeOpt,
                                                    UDate                date,
                                                    int32_t*             rawOffset,
                                                    int32_t*             dstOffset,
                                                    UErrorCode*          status);

#endif  // APPLE_ICU_CHANGES

#endif // UATIMEZONE_H
