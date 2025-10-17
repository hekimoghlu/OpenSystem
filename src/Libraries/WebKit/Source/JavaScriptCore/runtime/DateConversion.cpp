/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#include "config.h"
#include "DateConversion.h"

#include "JSDateMath.h"
#include <wtf/Assertions.h>
#include <wtf/DateMath.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template<int width>
static inline void appendNumber(StringBuilder& builder, int value)
{
    if (value < 0) {
        builder.append('-');
        value = -value;
    }
    String valueString = String::number(value);
    int fillingZerosCount = width - valueString.length();
    for (int i = 0; i < fillingZerosCount; ++i)
        builder.append('0');
    builder.append(valueString);
}

template<>
void appendNumber<2>(StringBuilder& builder, int value)
{
    ASSERT(0 <= value && value <= 99);
    builder.append(static_cast<char>('0' + value / 10));
    builder.append(static_cast<char>('0' + value % 10));
}

String formatDateTime(const GregorianDateTime& t, DateTimeFormat format, bool asUTCVariant, DateCache& dateCache)
{
    bool appendDate = format & DateTimeFormatDate;
    bool appendTime = format & DateTimeFormatTime;

    StringBuilder builder;

    if (appendDate) {
        builder.append(WTF::weekdayName[(t.weekDay() + 6) % 7]);

        if (asUTCVariant) {
            builder.append(", "_s);
            appendNumber<2>(builder, t.monthDay());
            builder.append(' ', WTF::monthName[t.month()]);
        } else {
            builder.append(' ', WTF::monthName[t.month()], ' ');
            appendNumber<2>(builder, t.monthDay());
        }
        builder.append(' ');
        appendNumber<4>(builder, t.year());
    }

    if (appendDate && appendTime)
        builder.append(' ');

    if (appendTime) {
        appendNumber<2>(builder, t.hour());
        builder.append(':');
        appendNumber<2>(builder, t.minute());
        builder.append(':');
        appendNumber<2>(builder, t.second());
        builder.append(" GMT"_s);

        if (!asUTCVariant) {
            int offset = std::abs(t.utcOffsetInMinute());
            builder.append(t.utcOffsetInMinute() < 0 ? '-' : '+');
            appendNumber<2>(builder, offset / 60);
            appendNumber<2>(builder, offset % 60);
            String timeZoneName = dateCache.timeZoneDisplayName(t.isDST());
            if (!timeZoneName.isEmpty())
                builder.append(" ("_s, timeZoneName, ')');
        }
    }

    return builder.toString().impl();
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
