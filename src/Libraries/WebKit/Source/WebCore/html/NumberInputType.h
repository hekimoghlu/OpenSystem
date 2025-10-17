/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include "TextFieldInputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class NumberInputType final : public TextFieldInputType {
    WTF_MAKE_TZONE_ALLOCATED(NumberInputType);
public:
    static Ref<NumberInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new NumberInputType(element));
    }

    bool typeMismatchFor(const String&) const final;
    bool typeMismatch() const final;
    bool hasBadInput() const final;

private:
    explicit NumberInputType(HTMLInputElement& element)
        : TextFieldInputType(Type::Number, element)
    {
    }

    const AtomString& formControlType() const final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;
    double valueAsDouble() const final;
    ExceptionOr<void> setValueAsDouble(double, TextFieldEventBehavior) const final;
    ExceptionOr<void> setValueAsDecimal(const Decimal&, TextFieldEventBehavior) const final;
    bool sizeShouldIncludeDecoration(int defaultSize, int& preferredSize) const final;
    float decorationWidth() const final;
    StepRange createStepRange(AnyStepHandling) const final;
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    Decimal parseToNumber(const String&, const Decimal&) const final;
    String serialize(const Decimal&) const final;
    String localizeValue(const String&) const final;
    String visibleValue() const final;
    String convertFromVisibleValue(const String&) const final;
    String sanitizeValue(const String&) const final;
    String badInputText() const final;
    bool supportsPlaceholder() const final;
    void attributeChanged(const QualifiedName&) final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(NumberInputType, Type::Number)
