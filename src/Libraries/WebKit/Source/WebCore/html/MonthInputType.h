/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include "BaseDateAndTimeInputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MonthInputType final : public BaseDateAndTimeInputType {
    WTF_MAKE_TZONE_ALLOCATED(MonthInputType);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MonthInputType);
public:
    static Ref<MonthInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new MonthInputType(element));
    }

private:
    explicit MonthInputType(HTMLInputElement& element)
        : BaseDateAndTimeInputType(Type::Month, element)
    {
    }

    const AtomString& formControlType() const final;
    DateComponentsType dateType() const final;
    WallTime valueAsDate() const final;
    String serializeWithMilliseconds(double) const final;
    Decimal parseToNumber(const String&, const Decimal&) const final;
    Decimal defaultValueForStepUp() const final;
    StepRange createStepRange(AnyStepHandling) const final;
    std::optional<DateComponents> parseToDateComponents(StringView) const final;
    std::optional<DateComponents> setMillisecondToDateComponents(double) const final;
    void handleDOMActivateEvent(Event&) final;
    void showPicker() final;

    bool isValidFormat(OptionSet<DateTimeFormatValidationResults>) const final;
    String formatDateTimeFieldsState(const DateTimeFieldsState&) const final;
    void setupLayoutParameters(DateTimeEditElement::LayoutParameters&, const DateComponents&) const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(MonthInputType, Type::Month)
