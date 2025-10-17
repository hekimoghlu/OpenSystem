/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#include "DateTimeFieldElement.h"

#include "CSSPropertyNames.h"
#include "DateComponents.h"
#include "EventNames.h"
#include "HTMLNames.h"
#include "KeyboardEvent.h"
#include "LocalizedStrings.h"
#include "PlatformLocale.h"
#include "RenderStyle.h"
#include "RenderTheme.h"
#include "ResolvedStyle.h"
#include "StyleResolver.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace HTMLNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DateTimeFieldElement);

DateTimeFieldElementFieldOwner::~DateTimeFieldElementFieldOwner() = default;

DateTimeFieldElement::DateTimeFieldElement(Document& document, DateTimeFieldElementFieldOwner& fieldOwner)
    : HTMLDivElement(divTag, document, TypeFlag::HasCustomStyleResolveCallbacks)
    , m_fieldOwner(fieldOwner)
{
}

std::optional<Style::ResolvedStyle> DateTimeFieldElement::resolveCustomStyle(const Style::ResolutionContext& resolutionContext, const RenderStyle* shadowHostStyle)
{
    auto elementStyle = resolveStyle(resolutionContext);

    adjustMinInlineSize(*elementStyle.style);

    if (!hasValue() && shadowHostStyle) {
        auto textColor = shadowHostStyle->visitedDependentColorWithColorFilter(CSSPropertyColor);
        auto backgroundColor = shadowHostStyle->visitedDependentColorWithColorFilter(CSSPropertyBackgroundColor);
        elementStyle.style->setColor(RenderTheme::singleton().datePlaceholderTextColor(textColor, backgroundColor));
    }

    return elementStyle;
}

void DateTimeFieldElement::defaultEventHandler(Event& event)
{
    if (event.type() == eventNames().blurEvent)
        handleBlurEvent(event);

    if (auto* keyboardEvent = dynamicDowncast<KeyboardEvent>(event)) {
        if (!isFieldOwnerDisabled() && !isFieldOwnerReadOnly()) {
            handleKeyboardEvent(*keyboardEvent);
            if (keyboardEvent->defaultHandled())
                return;
        }

        defaultKeyboardEventHandler(*keyboardEvent);
        if (keyboardEvent->defaultHandled())
            return;
    }

    HTMLDivElement::defaultEventHandler(event);
}

void DateTimeFieldElement::defaultKeyboardEventHandler(KeyboardEvent& keyboardEvent)
{
    if (isFieldOwnerDisabled())
        return;

    if (keyboardEvent.type() != eventNames().keydownEvent)
        return;

    auto key = keyboardEvent.keyIdentifier();
    auto code = keyboardEvent.code();

    bool isHorizontal = isFieldOwnerHorizontal();
    auto nextKeyIdentifier = isHorizontal ? "Right"_s : "Down"_s;
    auto previousKeyIdentifier = isHorizontal ? "Left"_s : "Up"_s;
    auto stepUpKeyIdentifier = isHorizontal ? "Up"_s : "Right"_s;
    auto stepDownKeyIdentifier = isHorizontal ? "Down"_s : "Left"_s;

    if (key == previousKeyIdentifier && m_fieldOwner && m_fieldOwner->focusOnPreviousField(*this)) {
        keyboardEvent.setDefaultHandled();
        return;
    }

    if ((key == nextKeyIdentifier || code == "Comma"_s || code == "Minus"_s || code == "Period"_s || code == "Slash"_s || code == "Semicolon"_s)
        && m_fieldOwner && m_fieldOwner->focusOnNextField(*this)) {
        keyboardEvent.setDefaultHandled();
        return;
    }

    if (isFieldOwnerReadOnly())
        return;

    if (key == stepUpKeyIdentifier) {
        stepUp();
        keyboardEvent.setDefaultHandled();
        return;
    }

    if (key == stepDownKeyIdentifier) {
        stepDown();
        keyboardEvent.setDefaultHandled();
        return;
    }

    // Clear value when pressing backspace or delete.
    if (key == "U+0008"_s || key == "U+007F"_s) {
        setEmptyValue(DispatchInputAndChangeEvents);
        keyboardEvent.setDefaultHandled();
        return;
    }
}

bool DateTimeFieldElement::isFieldOwnerDisabled() const
{
    return m_fieldOwner && m_fieldOwner->isFieldOwnerDisabled();
}

bool DateTimeFieldElement::isFieldOwnerReadOnly() const
{
    return m_fieldOwner && m_fieldOwner->isFieldOwnerReadOnly();
}

bool DateTimeFieldElement::isFieldOwnerHorizontal() const
{
    if (m_fieldOwner)
        return m_fieldOwner->isFieldOwnerHorizontal();
    return true;
}

bool DateTimeFieldElement::isFocusable() const
{
    if (isFieldOwnerDisabled())
        return false;
    return HTMLElement::isFocusable();
}

void DateTimeFieldElement::handleBlurEvent(Event& event)
{
    if (m_fieldOwner)
        m_fieldOwner->didBlurFromField(event);
}

Locale& DateTimeFieldElement::localeForOwner() const
{
    return document().getCachedLocale(localeIdentifier());
}

AtomString DateTimeFieldElement::localeIdentifier() const
{
    return m_fieldOwner ? m_fieldOwner->localeIdentifier() : nullAtom();
}

String DateTimeFieldElement::visibleValue() const
{
    return hasValue() ? value() : placeholderValue();
}

void DateTimeFieldElement::updateVisibleValue(EventBehavior eventBehavior)
{
    if (!firstChild())
        appendChild(Text::create(document(), String { emptyString() }));

    Ref textNode = downcast<Text>(*firstChild());
    String newVisibleValue = visibleValue();
    if (textNode->wholeText() != newVisibleValue)
        textNode->replaceWholeText(newVisibleValue);

    if (eventBehavior == DispatchInputAndChangeEvents && m_fieldOwner)
        m_fieldOwner->fieldValueChanged();
}

bool DateTimeFieldElement::supportsFocus() const
{
    return true;
}

} // namespace WebCore
