/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "MathMLElement.h"

#if ENABLE(MATHML)

#include "ElementInlines.h"
#include "EventHandler.h"
#include "FrameLoader.h"
#include "HTMLAnchorElement.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "HTMLTableCellElement.h"
#include "MathMLNames.h"
#include "MouseEvent.h"
#include "NodeName.h"
#include "RenderTableCell.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLElement);

using namespace MathMLNames;

MathMLElement::MathMLElement(const QualifiedName& tagName, Document& document, OptionSet<TypeFlag> typeFlags)
    : StyledElement(tagName, document, typeFlags | TypeFlag::IsMathMLElement)
{
}

Ref<MathMLElement> MathMLElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLElement(tagName, document));
}

unsigned MathMLElement::colSpan() const
{
    if (!hasTagName(mtdTag))
        return 1u;
    auto& colSpanValue = attributeWithoutSynchronization(columnspanAttr);
    return std::max(1u, limitToOnlyHTMLNonNegative(colSpanValue, 1u));
}

unsigned MathMLElement::rowSpan() const
{
    if (!hasTagName(mtdTag))
        return 1u;
    auto& rowSpanValue = attributeWithoutSynchronization(rowspanAttr);
    return std::max(1u, std::min(limitToOnlyHTMLNonNegative(rowSpanValue, 1u), HTMLTableCellElement::maxRowspan));
}

void MathMLElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::hrefAttr:
        setIsLink(!newValue.isNull() && !shouldProhibitLinks(this));
        break;
    case AttributeNames::columnspanAttr:
    case AttributeNames::rowspanAttr:
        if (is<RenderTableCell>(renderer()) && hasTagName(mtdTag))
            downcast<RenderTableCell>(*renderer()).colSpanOrRowSpanChanged();
        break;
    case AttributeNames::tabindexAttr:
        if (newValue.isEmpty())
            setTabIndexExplicitly(std::nullopt);
        else if (auto optionalTabIndex = parseHTMLInteger(newValue))
            setTabIndexExplicitly(optionalTabIndex.value());
        break;
    default:
        if (auto& eventName = HTMLElement::eventNameForEventHandlerAttribute(name); !eventName.isNull()) {
            setAttributeEventListener(eventName, name, newValue);
            return;
        }
        StyledElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
        break;
    }
}

bool MathMLElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::backgroundAttr:
    case AttributeNames::colorAttr:
    case AttributeNames::dirAttr:
    case AttributeNames::fontfamilyAttr:
    case AttributeNames::fontsizeAttr:
    case AttributeNames::fontstyleAttr:
    case AttributeNames::fontweightAttr:
    case AttributeNames::mathbackgroundAttr:
    case AttributeNames::mathcolorAttr:
    case AttributeNames::mathsizeAttr:
    case AttributeNames::displaystyleAttr:
        return true;
    default:
        break;
    }
    return StyledElement::hasPresentationalHintsForAttribute(name);
}

static inline bool isDisallowedMathSizeAttribute(const AtomString& value)
{
    // FIXME(https://webkit.org/b/245927): The CSS parser sometimes accept non-zero <number> font-size values on MathML elements, so explicitly disallow them.
    bool ok;
    value.toDouble(&ok);
    if (ok && value != "0"_s)
        return true;

    // Keywords from CSS font-size are disallowed.
    return equalIgnoringASCIICase(value, "medium"_s)
        || value.endsWithIgnoringASCIICase("large"_s)
        || value.endsWithIgnoringASCIICase("small"_s)
        || equalIgnoringASCIICase(value, "smaller"_s)
        || equalIgnoringASCIICase(value, "larger"_s)
        || equalIgnoringASCIICase(value, "math"_s);
}

static String convertMathSizeIfNeeded(const AtomString& value)
{
    if (value == "small"_s)
        return "0.75em"_s;
    if (value == "normal"_s)
        return "1em"_s;
    if (value == "big"_s)
        return "1.5em"_s;

    // FIXME: mathsize accepts any MathML length, including named spaces (see parseMathMLLength).
    // FIXME: Might be better to use double than float.
    // FIXME: Might be better to use "shortest" numeric formatting instead of fixed width.
    bool ok = false;
    float unitlessValue = value.toFloat(&ok);
    if (!ok)
        return value;
    return makeString(FormattedNumber::fixedWidth(unitlessValue * 100, 3), '%');
}

void MathMLElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::mathbackgroundAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyBackgroundColor, value);
        return;
    case AttributeNames::mathsizeAttr:
        if (document().settings().coreMathMLEnabled()) {
            if (!isDisallowedMathSizeAttribute(value))
                addPropertyToPresentationalHintStyle(style, CSSPropertyFontSize, value);
        } else
            addPropertyToPresentationalHintStyle(style, CSSPropertyFontSize, convertMathSizeIfNeeded(value));
        return;
    case AttributeNames::mathcolorAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyColor, value);
        return;
    case AttributeNames::dirAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyDirection, value);
        return;
    case AttributeNames::displaystyleAttr:
        if (equalLettersIgnoringASCIICase(value, "false"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyMathStyle, CSSValueCompact);
        else if (equalLettersIgnoringASCIICase(value, "true"_s))
            addPropertyToPresentationalHintStyle(style, CSSPropertyMathStyle, CSSValueNormal);
        return;
    default:
        break;
    }

    if (document().settings().coreMathMLEnabled()) {
        StyledElement::collectPresentationalHintsForAttribute(name, value, style);
        return;
    }

    // FIXME: The following are deprecated attributes that should lose if there is a conflict with a non-deprecated attribute.
    switch (name.nodeName()) {
    case AttributeNames::fontsizeAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyFontSize, value);
        break;
    case AttributeNames::backgroundAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyBackgroundColor, value);
        break;
    case AttributeNames::colorAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyColor, value);
        break;
    case AttributeNames::fontstyleAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyFontStyle, value);
        break;
    case AttributeNames::fontweightAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyFontWeight, value);
        break;
    case AttributeNames::fontfamilyAttr:
        addPropertyToPresentationalHintStyle(style, CSSPropertyFontFamily, value);
        break;
    default:
        ASSERT(!hasPresentationalHintsForAttribute(name));
        StyledElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

bool MathMLElement::childShouldCreateRenderer(const Node& child) const
{
    // In general, only MathML children are allowed. Text nodes are only visible in token MathML elements.
    return is<MathMLElement>(child);
}

bool MathMLElement::willRespondToMouseClickEventsWithEditability(Editability editability) const
{
    return isLink() || StyledElement::willRespondToMouseClickEventsWithEditability(editability);
}

void MathMLElement::defaultEventHandler(Event& event)
{
    if (isLink()) {
        if (focused() && isEnterKeyKeydownEvent(event)) {
            event.setDefaultHandled();
            dispatchSimulatedClick(&event);
            return;
        }
        if (MouseEvent::canTriggerActivationBehavior(event)) {
            const auto& href = attributeWithoutSynchronization(hrefAttr);
            event.setDefaultHandled();
            if (RefPtr frame = document().frame())
                frame->protectedLoader()->changeLocation(document().completeURL(href), selfTargetFrameName(), &event, ReferrerPolicy::EmptyString, document().shouldOpenExternalURLsPolicyToPropagate());
            return;
        }
    }

    StyledElement::defaultEventHandler(event);
}

bool MathMLElement::canStartSelection() const
{
    if (!isLink())
        return StyledElement::canStartSelection();

    return hasEditableStyle();
}

bool MathMLElement::isKeyboardFocusable(KeyboardEvent* event) const
{
    if (isFocusable() && StyledElement::supportsFocus())
        return StyledElement::isKeyboardFocusable(event);

    if (isLink())
        return document().frame()->eventHandler().tabsToLinks(event);

    return StyledElement::isKeyboardFocusable(event);
}

bool MathMLElement::isMouseFocusable() const
{
    // Links are focusable by default, but only allow links with tabindex or contenteditable to be mouse focusable.
    // https://bugs.webkit.org/show_bug.cgi?id=26856
    if (isLink())
        return StyledElement::supportsFocus();

    return StyledElement::isMouseFocusable();
}

bool MathMLElement::isURLAttribute(const Attribute& attribute) const
{
    return attribute.name().localName() == hrefAttr || StyledElement::isURLAttribute(attribute);
}

bool MathMLElement::supportsFocus() const
{
    if (hasEditableStyle())
        return StyledElement::supportsFocus();
    // If not a link we should still be able to focus the element if it has tabIndex.
    return isLink() || StyledElement::supportsFocus();
}

}

#endif // ENABLE(MATHML)
