/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#include "HTMLMarqueeElement.h"

#include "Attribute.h"
#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "NodeName.h"
#include "RenderLayer.h"
#include "RenderLayerScrollableArea.h"
#include "RenderMarquee.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLMarqueeElement);

using namespace HTMLNames;

inline HTMLMarqueeElement::HTMLMarqueeElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document, TypeFlag::HasDidMoveToNewDocument)
    , ActiveDOMObject(document)
{
    ASSERT(hasTagName(marqueeTag));
}

Ref<HTMLMarqueeElement> HTMLMarqueeElement::create(const QualifiedName& tagName, Document& document)
{
    auto marqueeElement = adoptRef(*new HTMLMarqueeElement(tagName, document));
    marqueeElement->suspendIfNeeded();
    return marqueeElement;
}

int HTMLMarqueeElement::minimumDelay() const
{
    if (!hasAttributeWithoutSynchronization(truespeedAttr)) {
        // WinIE uses 60ms as the minimum delay by default.
        return 60;
    }
    return 16; // Don't allow timers at < 16ms intervals to avoid CPU hogging: webkit.org/b/160609
}

bool HTMLMarqueeElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    switch (name.nodeName()) {
    case AttributeNames::widthAttr:
    case AttributeNames::heightAttr:
    case AttributeNames::bgcolorAttr:
    case AttributeNames::vspaceAttr:
    case AttributeNames::hspaceAttr:
    case AttributeNames::scrollamountAttr:
    case AttributeNames::scrolldelayAttr:
    case AttributeNames::loopAttr:
    case AttributeNames::behaviorAttr:
    case AttributeNames::directionAttr:
        return true;
    default:
        break;
    }
    return HTMLElement::hasPresentationalHintsForAttribute(name);
}

void HTMLMarqueeElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    switch (name.nodeName()) {
    case AttributeNames::widthAttr:
        if (!value.isEmpty())
            addHTMLLengthToStyle(style, CSSPropertyWidth, value);
        break;
    case AttributeNames::heightAttr:
        if (!value.isEmpty())
            addHTMLLengthToStyle(style, CSSPropertyHeight, value);
        break;
    case AttributeNames::bgcolorAttr:
        if (!value.isEmpty())
            addHTMLColorToStyle(style, CSSPropertyBackgroundColor, value);
        break;
    case AttributeNames::vspaceAttr:
        if (!value.isEmpty()) {
            addHTMLLengthToStyle(style, CSSPropertyMarginTop, value);
            addHTMLLengthToStyle(style, CSSPropertyMarginBottom, value);
        }
        break;
    case AttributeNames::hspaceAttr:
        if (!value.isEmpty()) {
            addHTMLLengthToStyle(style, CSSPropertyMarginLeft, value);
            addHTMLLengthToStyle(style, CSSPropertyMarginRight, value);
        }
        break;
    case AttributeNames::scrollamountAttr:
        if (!value.isEmpty())
            addHTMLLengthToStyle(style, CSSPropertyWebkitMarqueeIncrement, value);
        break;
    case AttributeNames::scrolldelayAttr:
        if (!value.isEmpty())
            addPropertyToPresentationalHintStyle(style, CSSPropertyWebkitMarqueeSpeed, limitToOnlyHTMLNonNegative(value, RenderStyle::initialMarqueeSpeed()), CSSUnitType::CSS_MS);
        break;
    case AttributeNames::loopAttr:
        if (!value.isEmpty()) {
            if (value == "-1"_s || equalLettersIgnoringASCIICase(value, "infinite"_s))
                addPropertyToPresentationalHintStyle(style, CSSPropertyWebkitMarqueeRepetition, CSSValueInfinite);
            else
                addHTMLNumberToStyle(style, CSSPropertyWebkitMarqueeRepetition, value);
        }
        break;
    case AttributeNames::behaviorAttr:
        if (!value.isEmpty())
            addPropertyToPresentationalHintStyle(style, CSSPropertyWebkitMarqueeStyle, value);
        break;
    case AttributeNames::directionAttr:
        if (!value.isEmpty())
            addPropertyToPresentationalHintStyle(style, CSSPropertyWebkitMarqueeDirection, value);
        break;
    default:
        HTMLElement::collectPresentationalHintsForAttribute(name, value, style);
        break;
    }
}

void HTMLMarqueeElement::start()
{
    if (auto* renderer = renderMarquee())
        renderer->start();
}

void HTMLMarqueeElement::stop()
{
    if (auto* renderer = renderMarquee())
        renderer->stop();
}

unsigned HTMLMarqueeElement::scrollAmount() const
{
    return limitToOnlyHTMLNonNegative(attributeWithoutSynchronization(scrollamountAttr), RenderStyle::initialMarqueeIncrement().intValue());
}
    
void HTMLMarqueeElement::setScrollAmount(unsigned scrollAmount)
{
    setUnsignedIntegralAttribute(scrollamountAttr, limitToOnlyHTMLNonNegative(scrollAmount, RenderStyle::initialMarqueeIncrement().intValue()));
}
    
unsigned HTMLMarqueeElement::scrollDelay() const
{
    return limitToOnlyHTMLNonNegative(attributeWithoutSynchronization(scrolldelayAttr), RenderStyle::initialMarqueeSpeed());
}

void HTMLMarqueeElement::setScrollDelay(unsigned scrollDelay)
{
    setUnsignedIntegralAttribute(scrolldelayAttr, limitToOnlyHTMLNonNegative(scrollDelay, RenderStyle::initialMarqueeSpeed()));
}
    
int HTMLMarqueeElement::loop() const
{
    int loopValue = getIntegralAttribute(loopAttr);
    return loopValue > 0 ? loopValue : -1;
}
    
ExceptionOr<void> HTMLMarqueeElement::setLoop(int loop)
{
    if (loop <= 0 && loop != -1)
        return Exception { ExceptionCode::IndexSizeError };
    setIntegralAttribute(loopAttr, loop);
    return { };
}

void HTMLMarqueeElement::didMoveToNewDocument(Document& oldDocument, Document& newDocument)
{
    HTMLElement::didMoveToNewDocument(oldDocument, newDocument);
    ActiveDOMObject::didMoveToNewDocument(newDocument);
}

void HTMLMarqueeElement::suspend(ReasonForSuspension)
{
    if (RenderMarquee* marqueeRenderer = renderMarquee())
        marqueeRenderer->suspend();
}

void HTMLMarqueeElement::resume()
{
    if (RenderMarquee* marqueeRenderer = renderMarquee())
        marqueeRenderer->updateMarqueePosition();
}

RenderMarquee* HTMLMarqueeElement::renderMarquee() const
{
    if (!renderer() || !renderer()->hasLayer())
        return nullptr;
    auto* scrollableArea = renderBoxModelObject()->layer()->scrollableArea();
    if (!scrollableArea)
        return nullptr;
    return scrollableArea->marquee();
}

} // namespace WebCore
