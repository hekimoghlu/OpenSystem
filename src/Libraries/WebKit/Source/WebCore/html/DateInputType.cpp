/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "DateInputType.h"

#include "DateComponents.h"
#include "DateTimeFieldsState.h"
#include "ElementInlines.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "InputTypeNames.h"
#include "PlatformLocale.h"
#include "StepRange.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DateInputType);

using namespace HTMLNames;

static const int dateDefaultStep = 1;
static const int dateDefaultStepBase = 0;
static const int dateStepScaleFactor = 86400000;
static const StepRange::StepDescription dateStepDescription { dateDefaultStep, dateDefaultStepBase, dateStepScaleFactor, StepRange::ParsedStepValueShouldBeInteger };

DateInputType::DateInputType(HTMLInputElement& element)
    : BaseDateAndTimeInputType(Type::Date, element)
{
}

const AtomString& DateInputType::formControlType() const
{
    return InputTypeNames::date();
}

DateComponentsType DateInputType::dateType() const
{
    return DateComponentsType::Date;
}

StepRange DateInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(dateDefaultStepBase);
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), Decimal::fromDouble(DateComponents::minimumDate()));
    const Decimal maximum = parseToNumber(element()->attributeWithoutSynchronization(maxAttr), Decimal::fromDouble(DateComponents::maximumDate()));
    const Decimal step = StepRange::parseStep(anyStepHandling, dateStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, dateStepDescription);
}

std::optional<DateComponents> DateInputType::parseToDateComponents(StringView source) const
{
    return DateComponents::fromParsingDate(source);
}

std::optional<DateComponents> DateInputType::setMillisecondToDateComponents(double value) const
{
    return DateComponents::fromMillisecondsSinceEpochForDate(value);
}

bool DateInputType::isValidFormat(OptionSet<DateTimeFormatValidationResults> results) const
{
    return results.containsAll({ DateTimeFormatValidationResults::HasYear, DateTimeFormatValidationResults::HasMonth, DateTimeFormatValidationResults::HasDay });
}

String DateInputType::formatDateTimeFieldsState(const DateTimeFieldsState& state) const
{
    if (!state.dayOfMonth || !state.month || !state.year)
        return emptyString();

    return makeString(pad('0', 4, *state.year), '-', pad('0', 2, *state.month), '-', pad('0', 2, *state.dayOfMonth));
}

void DateInputType::setupLayoutParameters(DateTimeEditElement::LayoutParameters& layoutParameters, const DateComponents&) const
{
    layoutParameters.dateTimeFormat = layoutParameters.locale.dateFormat();
    layoutParameters.fallbackDateTimeFormat = "yyyy-MM-dd"_s;
}

} // namespace WebCore
