/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#include "MonthInputType.h"

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
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MonthInputType);

using namespace HTMLNames;

static const int monthDefaultStep = 1;
static const int monthDefaultStepBase = 0;
static const int monthStepScaleFactor = 1;
static const StepRange::StepDescription monthStepDescription { monthDefaultStep, monthDefaultStepBase, monthStepScaleFactor, StepRange::ParsedStepValueShouldBeInteger };

const AtomString& MonthInputType::formControlType() const
{
    return InputTypeNames::month();
}

DateComponentsType MonthInputType::dateType() const
{
    return DateComponentsType::Month;
}

WallTime MonthInputType::valueAsDate() const
{
    ASSERT(element());
    auto date = parseToDateComponents(element()->value());
    if (!date)
        return WallTime::nan();
    double msec = date->millisecondsSinceEpoch();
    ASSERT(std::isfinite(msec));
    return WallTime::fromRawSeconds(Seconds::fromMilliseconds(msec).value());
}

String MonthInputType::serializeWithMilliseconds(double value) const
{
    auto date = DateComponents::fromMillisecondsSinceEpochForMonth(value);
    if (!date)
        return { };
    return serializeWithComponents(*date);
}

Decimal MonthInputType::defaultValueForStepUp() const
{
    double current = WallTime::now().secondsSinceEpoch().milliseconds();
    int offset = calculateLocalTimeOffset(current).offset / msPerMinute;
    current += offset * msPerMinute;

    auto date = DateComponents::fromMillisecondsSinceEpochForMonth(current);
    if (!date)
        return  { };

    double months = date->monthsSinceEpoch();
    ASSERT(std::isfinite(months));
    return Decimal::fromDouble(months);
}

StepRange MonthInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(Decimal::fromDouble(monthDefaultStepBase));
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), Decimal::fromDouble(DateComponents::minimumMonth()));
    const Decimal maximum = parseToNumber(element()->attributeWithoutSynchronization(maxAttr), Decimal::fromDouble(DateComponents::maximumMonth()));
    const Decimal step = StepRange::parseStep(anyStepHandling, monthStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, monthStepDescription);
}

Decimal MonthInputType::parseToNumber(const String& src, const Decimal& defaultValue) const
{
    auto date = parseToDateComponents(src);
    if (!date)
        return defaultValue;
    double months = date->monthsSinceEpoch();
    ASSERT(std::isfinite(months));
    return Decimal::fromDouble(months);
}

std::optional<DateComponents> MonthInputType::parseToDateComponents(StringView source) const
{
    return DateComponents::fromParsingMonth(source);
}

std::optional<DateComponents> MonthInputType::setMillisecondToDateComponents(double value) const
{
    return DateComponents::fromMonthsSinceEpoch(value);
}

void MonthInputType::handleDOMActivateEvent(Event&)
{
}

void MonthInputType::showPicker()
{
}

bool MonthInputType::isValidFormat(OptionSet<DateTimeFormatValidationResults> results) const
{
    return results.containsAll({ DateTimeFormatValidationResults::HasYear, DateTimeFormatValidationResults::HasMonth });
}

String MonthInputType::formatDateTimeFieldsState(const DateTimeFieldsState& state) const
{
    if (!state.year || !state.month)
        return emptyString();

    return makeString(pad('0', 4, *state.year), '-', pad('0', 2, *state.month));
}

void MonthInputType::setupLayoutParameters(DateTimeEditElement::LayoutParameters& layoutParameters, const DateComponents&) const
{
    layoutParameters.dateTimeFormat = layoutParameters.locale.shortMonthFormat();
    layoutParameters.fallbackDateTimeFormat = "yyyy-MM"_s;
}

} // namespace WebCore
