/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#pragma once

#include <wtf/Forward.h>

namespace WebCore {

// DateTimeFormat parses date time format defined in Unicode Technical
// standard 35, Locale Data Markup Language (LDML)[1].
// [1] LDML http://unicode.org/reports/tr35/tr35-6.html#Date_Format_Patterns
class DateTimeFormat {
public:
    enum FieldType {
        FieldTypeInvalid,
        FieldTypeLiteral,

        // Era: AD
        FieldTypeEra = 'G',

        // Year: 1996
        FieldTypeYear = 'y',
        FieldTypeYearOfWeekOfYear = 'Y',
        FieldTypeExtendedYear = 'u',

        // Quater: Q2
        FieldTypeQuater = 'Q',
        FieldTypeQuaterStandAlone = 'q',

        // Month: September
        FieldTypeMonth = 'M',
        FieldTypeMonthStandAlone = 'L',

        // Week: 42
        FieldTypeWeekOfYear = 'w',
        FieldTypeWeekOfMonth = 'W',

        // Day: 12
        FieldTypeDayOfMonth = 'd',
        FieldTypeDayOfYear = 'D',
        FieldTypeDayOfWeekInMonth = 'F',
        FieldTypeModifiedJulianDay = 'g',

        // Week Day: Tuesday
        FieldTypeDayOfWeek = 'E',
        FieldTypeLocalDayOfWeek = 'e',
        FieldTypeLocalDayOfWeekStandAlon = 'c',

        // Period: AM or PM
        FieldTypePeriod = 'a',

        // Hour: 7
        FieldTypeHour12 = 'h',
        FieldTypeHour23 = 'H',
        FieldTypeHour11 = 'K',
        FieldTypeHour24 = 'k',

        // Minute: 59
        FieldTypeMinute = 'm',

        // Second: 12
        FieldTypeSecond = 's',
        FieldTypeFractionalSecond = 'S',
        FieldTypeMillisecondsInDay = 'A',

        // Zone: PDT
        FieldTypeZone = 'z',
        FieldTypeRFC822Zone = 'Z',
        FieldTypeNonLocationZone = 'v',
    };

    class TokenHandler {
    public:
        virtual ~TokenHandler() = default;
        virtual void visitField(FieldType, int numberOfPatternCharacters) = 0;
        virtual void visitLiteral(String&&) = 0;
    };

    // Returns true if succeeded, false if failed.
    static bool parse(const String&, TokenHandler&);
    static void quoteAndAppendLiteral(const String&, StringBuilder&);
};

} // namespace WebCore
