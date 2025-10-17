/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#include "DateTimeLocalInputType.h"

#include "DateComponents.h"
#include "DateTimeFieldsState.h"
#include "Decimal.h"
#include "ElementInlines.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "InputTypeNames.h"
#include "PlatformLocale.h"
#include "StepRange.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DateTimeLocalInputType);

using namespace HTMLNames;

static const int dateTimeLocalDefaultStep = 60;
static const int dateTimeLocalDefaultStepBase = 0;
static const int dateTimeLocalStepScaleFactor = 1000;
static const StepRange::StepDescription dateTimeLocalStepDescription { dateTimeLocalDefaultStep, dateTimeLocalDefaultStepBase, dateTimeLocalStepScaleFactor, StepRange::ScaledStepValueShouldBeInteger };

const AtomString& DateTimeLocalInputType::formControlType() const
{
    return InputTypeNames::datetimelocal();
}

DateComponentsType DateTimeLocalInputType::dateType() const
{
    return DateComponentsType::DateTimeLocal;
}

WallTime DateTimeLocalInputType::valueAsDate() const
{
    // valueAsDate doesn't work for the datetime-local type according to the standard.
    return WallTime::nan();
}

ExceptionOr<void> DateTimeLocalInputType::setValueAsDate(WallTime value) const
{
    // valueAsDate doesn't work for the datetime-local type according to the standard.
    return InputType::setValueAsDate(value);
}

StepRange DateTimeLocalInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(dateTimeLocalDefaultStepBase);
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), Decimal::fromDouble(DateComponents::minimumDateTime()));
    const Decimal maximum = parseToNumber(element()->attributeWithoutSynchronization(maxAttr), Decimal::fromDouble(DateComponents::maximumDateTime()));
    const Decimal step = StepRange::parseStep(anyStepHandling, dateTimeLocalStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, dateTimeLocalStepDescription);
}

std::optional<DateComponents> DateTimeLocalInputType::parseToDateComponents(StringView source) const
{
    return DateComponents::fromParsingDateTimeLocal(source);
}

std::optional<DateComponents> DateTimeLocalInputType::setMillisecondToDateComponents(double value) const
{
    return DateComponents::fromMillisecondsSinceEpochForDateTimeLocal(value);
}

bool DateTimeLocalInputType::isValidFormat(OptionSet<DateTimeFormatValidationResults> results) const
{
    return results.containsAll({ DateTimeFormatValidationResults::HasYear, DateTimeFormatValidationResults::HasMonth, DateTimeFormatValidationResults::HasDay, DateTimeFormatValidationResults::HasHour, DateTimeFormatValidationResults::HasMinute, DateTimeFormatValidationResults::HasMeridiem });
}

String DateTimeLocalInputType::sanitizeValue(const String& proposedValue) const
{
    if (proposedValue.isEmpty())
        return proposedValue;

    auto components = DateComponents::fromParsingDateTimeLocal(proposedValue);
    return components ? components->toString() : emptyString();
}

String DateTimeLocalInputType::formatDateTimeFieldsState(const DateTimeFieldsState& state) const
{
    if (!state.year || !state.month || !state.dayOfMonth || !state.hour || !state.minute || !state.meridiem)
        return emptyString();

    auto dateString = makeString(pad('0', 4, *state.year), '-', pad('0', 2, *state.month), '-', pad('0', 2, *state.dayOfMonth));
    auto hourMinuteString = makeString(pad('0', 2, *state.hour23()), ':', pad('0', 2, *state.minute));

    if (state.millisecond)
        return makeString(dateString, 'T', hourMinuteString, ':', pad('0', 2, state.second ? *state.second : 0), '.', pad('0', 3, *state.millisecond));

    if (state.second)
        return makeString(dateString, 'T', hourMinuteString, ':', pad('0', 2, *state.second));

    return makeString(dateString, 'T', hourMinuteString);
}

void DateTimeLocalInputType::setupLayoutParameters(DateTimeEditElement::LayoutParameters& layoutParameters, const DateComponents& date) const
{
    layoutParameters.shouldHaveMillisecondField = shouldHaveMillisecondField(date);

    if (layoutParameters.shouldHaveMillisecondField || shouldHaveSecondField(date)) {
        layoutParameters.dateTimeFormat = layoutParameters.locale.dateTimeFormatWithSeconds();
        layoutParameters.fallbackDateTimeFormat = "yyyy-MM-dd'T'HH:mm:ss"_s;
    } else {
        layoutParameters.dateTimeFormat = layoutParameters.locale.dateTimeFormatWithoutSeconds();
        layoutParameters.fallbackDateTimeFormat = "yyyy-MM-dd'T'HH:mm"_s;
    }
}

} // namespace WebCore
