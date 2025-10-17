/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "WeekInputType.h"

#include "DateComponents.h"
#include "Decimal.h"
#include "ElementInlines.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "InputTypeNames.h"
#include "StepRange.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WeekInputType);

using namespace HTMLNames;

static const int weekDefaultStepBase = -259200000; // The first day of 1970-W01.
static const int weekDefaultStep = 1;
static const int weekStepScaleFactor = 604800000;
static const StepRange::StepDescription weekStepDescription { weekDefaultStep, weekDefaultStepBase, weekStepScaleFactor, StepRange::ParsedStepValueShouldBeInteger };

const AtomString& WeekInputType::formControlType() const
{
    return InputTypeNames::week();
}

DateComponentsType WeekInputType::dateType() const
{
    return DateComponentsType::Week;
}

StepRange WeekInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(weekDefaultStepBase);
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), Decimal::fromDouble(DateComponents::minimumWeek()));
    const Decimal maximum = parseToNumber(element()->attributeWithoutSynchronization(maxAttr), Decimal::fromDouble(DateComponents::maximumWeek()));
    const Decimal step = StepRange::parseStep(anyStepHandling, weekStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, weekStepDescription);
}

std::optional<DateComponents> WeekInputType::parseToDateComponents(StringView source) const
{
    return DateComponents::fromParsingWeek(source);
}

std::optional<DateComponents> WeekInputType::setMillisecondToDateComponents(double value) const
{
    return DateComponents::fromMillisecondsSinceEpochForWeek(value);
}

void WeekInputType::handleDOMActivateEvent(Event&)
{
}

void WeekInputType::showPicker()
{
}

bool WeekInputType::isValidFormat(OptionSet<DateTimeFormatValidationResults> results) const
{
    return results.containsAll({ DateTimeFormatValidationResults::HasYear, DateTimeFormatValidationResults::HasWeek });
}

String WeekInputType::formatDateTimeFieldsState(const DateTimeFieldsState&) const
{
    return emptyString();
}

void WeekInputType::setupLayoutParameters(DateTimeEditElement::LayoutParameters&, const DateComponents&) const
{
}

} // namespace WebCore
