/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#import "config.h"
#import "CoreIPCDateComponents.h"

namespace WebKit {

std::array calendarUnitForComponentIndex {
    NSCalendarUnitEra,
    NSCalendarUnitYear,
    NSCalendarUnitYearForWeekOfYear,
    NSCalendarUnitQuarter,
    NSCalendarUnitMonth,
    NSCalendarUnitHour,
    NSCalendarUnitMinute,
    NSCalendarUnitSecond,
    NSCalendarUnitNanosecond,
    NSCalendarUnitWeekOfYear,
    NSCalendarUnitWeekOfMonth,
    NSCalendarUnitWeekday,
    NSCalendarUnitWeekdayOrdinal,
    NSCalendarUnitDay
};
static size_t numberOfComponentIndexes = sizeof(calendarUnitForComponentIndex) / sizeof(NSUInteger);

CoreIPCDateComponents::CoreIPCDateComponents(NSDateComponents *components)
{
    if (components.calendar)
        m_calendarIdentifier = components.calendar.calendarIdentifier;
    if (components.timeZone)
        m_timeZoneName = components.timeZone.name;

    m_componentValues.reserveInitialCapacity(numberOfComponentIndexes);
    for (size_t i = 0; i < numberOfComponentIndexes; ++i)
        m_componentValues.append([components valueForComponent:calendarUnitForComponentIndex[i]]);
}

RetainPtr<id> CoreIPCDateComponents::toID() const
{
    RetainPtr<NSDateComponents> components = adoptNS([NSDateComponents new]);

    for (size_t i = 0; i < numberOfComponentIndexes; ++i)
        [components setValue:m_componentValues[i] forComponent:calendarUnitForComponentIndex[i]];

    if (!m_calendarIdentifier.isEmpty())
        components.get().calendar = [NSCalendar calendarWithIdentifier:(NSString *)m_calendarIdentifier];
    if (!m_timeZoneName.isEmpty())
        components.get().timeZone = [NSTimeZone timeZoneWithName:(NSString *)m_timeZoneName];

    return components;
}

bool CoreIPCDateComponents::hasCorrectNumberOfComponentValues(const Vector<NSInteger>& componentValues)
{
    return componentValues.size() == numberOfComponentIndexes;
}

} // namespace WebKit
