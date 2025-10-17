/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include "TimeInputType.h"

#include "DateComponents.h"
#include "DateTimeFieldsState.h"
#include "Decimal.h"
#include "ElementInlines.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "InputTypeNames.h"
#include "PlatformLocale.h"
#include "StepRange.h"
#include <wtf/DateMath.h>
#include <wtf/MathExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TimeInputType);

using namespace HTMLNames;

static const int timeDefaultStep = 60;
static const int timeDefaultStepBase = 0;
static const int timeStepScaleFactor = 1000;
static const StepRange::StepDescription timeStepDescription { timeDefaultStep, timeDefaultStepBase, timeStepScaleFactor, StepRange::ScaledStepValueShouldBeInteger };

TimeInputType::TimeInputType(HTMLInputElement& element)
    : BaseDateAndTimeInputType(Type::Time, element)
{
}

const AtomString& TimeInputType::formControlType() const
{
    return InputTypeNames::time();
}

DateComponentsType TimeInputType::dateType() const
{
    return DateComponentsType::Time;
}

Decimal TimeInputType::defaultValueForStepUp() const
{
    double current = WallTime::now().secondsSinceEpoch().milliseconds();
    int offset = calculateLocalTimeOffset(current).offset / msPerMinute;
    current += offset * msPerMinute;

    auto date = DateComponents::fromMillisecondsSinceMidnight(current);
    if (!date)
        return  { };

    double milliseconds = date->millisecondsSinceEpoch();
    ASSERT(std::isfinite(milliseconds));
    return Decimal::fromDouble(milliseconds);
}

StepRange TimeInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(timeDefaultStepBase);
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), Decimal::fromDouble(DateComponents::minimumTime()));
    const Decimal maximum = parseToNumber(element()->attributeWithoutSynchronization(maxAttr), Decimal::fromDouble(DateComponents::maximumTime()));
    const Decimal step = StepRange::parseStep(anyStepHandling, timeStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, timeStepDescription, StepRange::IsReversible::Yes);
}

std::optional<DateComponents> TimeInputType::parseToDateComponents(StringView source) const
{
    return DateComponents::fromParsingTime(source);
}

std::optional<DateComponents> TimeInputType::setMillisecondToDateComponents(double value) const
{
    return DateComponents::fromMillisecondsSinceMidnight(value);
}

void TimeInputType::handleDOMActivateEvent(Event&)
{
}

void TimeInputType::showPicker()
{
}

bool TimeInputType::isValidFormat(OptionSet<DateTimeFormatValidationResults> results) const
{
    return results.containsAll({ DateTimeFormatValidationResults::HasHour, DateTimeFormatValidationResults::HasMinute, DateTimeFormatValidationResults::HasMeridiem });
}

String TimeInputType::formatDateTimeFieldsState(const DateTimeFieldsState& state) const
{
    if (!state.hour || !state.minute || !state.meridiem)
        return emptyString();

    auto hourMinuteString = makeString(pad('0', 2, *state.hour23()), ':', pad('0', 2, *state.minute));

    if (state.millisecond)
        return makeString(hourMinuteString, ':', pad('0', 2, state.second ? *state.second : 0), '.', pad('0', 3, *state.millisecond));

    if (state.second)
        return makeString(hourMinuteString, ':', pad('0', 2, *state.second));

    return hourMinuteString;
}

void TimeInputType::setupLayoutParameters(DateTimeEditElement::LayoutParameters& layoutParameters, const DateComponents& date) const
{
    layoutParameters.shouldHaveMillisecondField = shouldHaveMillisecondField(date);

    if (layoutParameters.shouldHaveMillisecondField || shouldHaveSecondField(date)) {
        layoutParameters.dateTimeFormat = layoutParameters.locale.timeFormat();
        layoutParameters.fallbackDateTimeFormat = "HH:mm:ss"_s;
    } else {
        layoutParameters.dateTimeFormat = layoutParameters.locale.shortTimeFormat();
        layoutParameters.fallbackDateTimeFormat = "HH:mm"_s;
    }
}

} // namespace WebCore
