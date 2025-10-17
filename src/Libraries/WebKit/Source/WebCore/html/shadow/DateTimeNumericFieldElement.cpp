/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
#include "DateTimeNumericFieldElement.h"

#include "EventNames.h"
#include "FontCascade.h"
#include "HTMLNames.h"
#include "KeyboardEvent.h"
#include "PlatformLocale.h"
#include "RenderBlock.h"
#include "RenderStyleSetters.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

constexpr Seconds typeAheadTimeout { 1_s };

int DateTimeNumericFieldElement::Range::clampValue(int value) const
{
    return std::clamp(value, minimum, maximum);
}

bool DateTimeNumericFieldElement::Range::isInRange(int value) const
{
    return value >= minimum && value <= maximum;
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DateTimeNumericFieldElement);

DateTimeNumericFieldElement::DateTimeNumericFieldElement(Document& document, DateTimeFieldElementFieldOwner& fieldOwner, const Range& range, int placeholder)
    : DateTimeFieldElement(document, fieldOwner)
    , m_range(range)
    , m_placeholder(formatValue(placeholder))
    , m_placeholderValue(placeholder)
{
}

void DateTimeNumericFieldElement::adjustMinInlineSize(RenderStyle& style) const
{
    auto& font = style.fontCascade();

    unsigned length = 2;
    if (m_range.maximum > 999)
        length = 4;
    else if (m_range.maximum > 99)
        length = 3;

    auto& locale = localeForOwner();

    float inlineSize = 0;
    for (char c = '0'; c <= '9'; ++c) {
        auto numberString = locale.convertToLocalizedNumber(makeString(pad(c, length, makeString(c))));
        inlineSize = std::max(inlineSize, font.width(RenderBlock::constructTextRun(numberString, style)));
    }

    if (style.writingMode().isHorizontal())
        style.setMinWidth({ inlineSize, LengthType::Fixed });
    else
        style.setMinHeight({ inlineSize, LengthType::Fixed });
}

int DateTimeNumericFieldElement::maximum() const
{
    return m_range.maximum;
}

String DateTimeNumericFieldElement::formatValue(int value) const
{
    Locale& locale = localeForOwner();

    if (m_range.maximum > 999)
        return locale.convertToLocalizedNumber(makeString(pad('0', 4, value)));
    if (m_range.maximum > 99)
        return locale.convertToLocalizedNumber(makeString(pad('0', 3, value)));
    return locale.convertToLocalizedNumber(makeString(pad('0', 2, value)));
}

bool DateTimeNumericFieldElement::hasValue() const
{
    return m_hasValue;
}

void DateTimeNumericFieldElement::setEmptyValue(EventBehavior eventBehavior)
{
    m_value = 0;
    m_hasValue = false;
    m_typeAheadBuffer.clear();
    updateVisibleValue(eventBehavior);
    setARIAValueAttributesWithInteger(0);
}

void DateTimeNumericFieldElement::setValueAsInteger(int value, EventBehavior eventBehavior)
{
    m_value = m_range.clampValue(value);
    m_hasValue = true;
    updateVisibleValue(eventBehavior);
    setARIAValueAttributesWithInteger(value);
}

void DateTimeNumericFieldElement::setValueAsIntegerByStepping(int value)
{
    m_typeAheadBuffer.clear();
    setValueAsInteger(value, DispatchInputAndChangeEvents);
}

void DateTimeNumericFieldElement::setARIAValueAttributesWithInteger(int value)
{
    setAttributeWithoutSynchronization(HTMLNames::aria_valuenowAttr, AtomString::number(value));
    setAttributeWithoutSynchronization(HTMLNames::aria_valuetextAttr, AtomString::number(value));
}

void DateTimeNumericFieldElement::stepDown()
{
    int newValue = m_hasValue ? m_value - 1 : m_range.maximum;
    if (!m_range.isInRange(newValue))
        newValue = m_range.maximum;
    setValueAsIntegerByStepping(newValue);
}

void DateTimeNumericFieldElement::stepUp()
{
    int newValue = m_hasValue ? m_value + 1 : m_range.minimum;
    if (!m_range.isInRange(newValue))
        newValue = m_range.minimum;
    setValueAsIntegerByStepping(newValue);
}

String DateTimeNumericFieldElement::value() const
{
    return m_hasValue ? formatValue(m_value) : emptyString();
}

String DateTimeNumericFieldElement::placeholderValue() const
{
    return m_placeholder;
}

void DateTimeNumericFieldElement::handleKeyboardEvent(KeyboardEvent& keyboardEvent)
{
    if (keyboardEvent.type() != eventNames().keypressEvent)
        return;

    auto charCode = static_cast<UChar>(keyboardEvent.charCode());
    String number = localeForOwner().convertFromLocalizedNumber(span(charCode));
    int digit = number[0] - '0';
    if (digit < 0 || digit > 9)
        return;

    Seconds timeSinceLastDigitChar = keyboardEvent.timeStamp() - m_lastDigitCharTime;
    m_lastDigitCharTime = keyboardEvent.timeStamp();

    if (timeSinceLastDigitChar > typeAheadTimeout) {
        m_typeAheadBuffer.clear();
    } else if (auto length = m_typeAheadBuffer.length()) {
        unsigned maxLength = formatValue(m_range.maximum).length();
        if (length == maxLength)
            m_typeAheadBuffer.clear();
    }

    m_typeAheadBuffer.append(number);
    setValueAsInteger(parseIntegerAllowingTrailingJunk<int>(m_typeAheadBuffer).value_or(0), DispatchInputAndChangeEvents);

    keyboardEvent.setDefaultHandled();
}

void DateTimeNumericFieldElement::handleBlurEvent(Event& event)
{
    m_typeAheadBuffer.clear();
    DateTimeFieldElement::handleBlurEvent(event);
}

} // namespace WebCore
