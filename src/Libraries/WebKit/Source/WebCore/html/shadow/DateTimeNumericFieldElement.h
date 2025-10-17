/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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

#include "DateTimeFieldElement.h"
#include <wtf/MonotonicTime.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class DateTimeNumericFieldElement : public DateTimeFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeNumericFieldElement);
public:
    struct Range {
        Range(int minimum, int maximum)
            : minimum(minimum), maximum(maximum) { }
        int clampValue(int) const;
        bool isInRange(int) const;

        int minimum;
        int maximum;
    };

protected:
    DateTimeNumericFieldElement(Document&, DateTimeFieldElementFieldOwner&, const Range&, int placeholder);

    int maximum() const;

    // DateTimeFieldElement functions:
    bool hasValue() const final;
    void setEmptyValue(EventBehavior = DispatchNoEvent) final;
    void setValueAsInteger(int, EventBehavior = DispatchNoEvent) final;
    void stepDown() final;
    void stepUp() final;
    int valueAsInteger() const final { return m_hasValue ? m_value : -1; }
    int placeholderValueAsInteger() const final { return m_placeholderValue; }


private:
    // DateTimeFieldElement functions:
    void adjustMinInlineSize(RenderStyle&) const final;
    String value() const final;
    String placeholderValue() const final;
    void handleKeyboardEvent(KeyboardEvent&) final;
    void handleBlurEvent(Event&) final;

    String formatValue(int) const;
    void setValueAsIntegerByStepping(int);
    void setARIAValueAttributesWithInteger(int);

    const Range m_range;
    const String m_placeholder;
    int m_placeholderValue { 0 };
    int m_value { 0 };
    bool m_hasValue { false };
    StringBuilder m_typeAheadBuffer;
    MonotonicTime m_lastDigitCharTime;
};

} // namespace WebCore
