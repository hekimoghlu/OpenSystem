/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include "DateTimeSymbolicFieldElement.h"

#include "EventNames.h"
#include "FontCascade.h"
#include "KeyboardEvent.h"
#include "RenderBlock.h"
#include "RenderStyleInlines.h"
#include "RenderStyleSetters.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextBreakIterator.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DateTimeSymbolicFieldElement);

DateTimeSymbolicFieldElement::DateTimeSymbolicFieldElement(Document& document, DateTimeFieldElementFieldOwner& fieldOwner, const Vector<String>& symbols, int placeholderIndex)
    : DateTimeFieldElement(document, fieldOwner)
    , m_symbols(symbols)
    , m_placeholderIndex(placeholderIndex)
    , m_typeAhead(this)
{
    ASSERT(!m_symbols.isEmpty());
}

void DateTimeSymbolicFieldElement::adjustMinInlineSize(RenderStyle& style) const
{
    auto& font = style.fontCascade();

    float inlineSize = 0;
    for (auto& symbol : m_symbols)
        inlineSize = std::max(inlineSize, font.width(RenderBlock::constructTextRun(symbol, style)));

    if (style.writingMode().isHorizontal())
        style.setMinWidth({ inlineSize, LengthType::Fixed });
    else
        style.setMinHeight({ inlineSize, LengthType::Fixed });
}

bool DateTimeSymbolicFieldElement::hasValue() const
{
    return m_selectedIndex >= 0;
}

void DateTimeSymbolicFieldElement::setEmptyValue(EventBehavior eventBehavior)
{
    m_selectedIndex = invalidIndex;
    updateVisibleValue(eventBehavior);
}

void DateTimeSymbolicFieldElement::setValueAsInteger(int newSelectedIndex, EventBehavior eventBehavior)
{
    m_selectedIndex = std::max(0, std::min(newSelectedIndex, static_cast<int>(m_symbols.size() - 1)));
    updateVisibleValue(eventBehavior);
}

void DateTimeSymbolicFieldElement::stepDown()
{
    int newValue = hasValue() ? m_selectedIndex - 1 : m_symbols.size() - 1;
    if (newValue < 0)
        newValue = m_symbols.size() - 1;
    setValueAsInteger(newValue, DispatchInputAndChangeEvents);
}

void DateTimeSymbolicFieldElement::stepUp()
{
    int newValue = hasValue() ? m_selectedIndex + 1 : 0;
    if (newValue >= static_cast<int>(m_symbols.size()))
        newValue = 0;
    setValueAsInteger(newValue, DispatchInputAndChangeEvents);
}

String DateTimeSymbolicFieldElement::value() const
{
    return hasValue() ? m_symbols[m_selectedIndex] : emptyString();
}

String DateTimeSymbolicFieldElement::placeholderValue() const
{
    return m_symbols[m_placeholderIndex];
}

void DateTimeSymbolicFieldElement::handleKeyboardEvent(KeyboardEvent& keyboardEvent)
{
    if (keyboardEvent.type() != eventNames().keypressEvent)
        return;

    keyboardEvent.setDefaultHandled();

    int index = m_typeAhead.handleEvent(&keyboardEvent, TypeAhead::MatchPrefix | TypeAhead::CycleFirstChar | TypeAhead::MatchIndex);
    if (index < 0)
        return;
    setValueAsInteger(index, DispatchInputAndChangeEvents);
}

int DateTimeSymbolicFieldElement::indexOfSelectedOption() const
{
    return m_selectedIndex;
}

int DateTimeSymbolicFieldElement::optionCount() const
{
    return m_symbols.size();
}

String DateTimeSymbolicFieldElement::optionAtIndex(int index) const
{
    return m_symbols[index];
}

} // namespace WebCore
