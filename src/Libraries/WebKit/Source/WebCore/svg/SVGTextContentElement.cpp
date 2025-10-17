/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#include "SVGTextContentElement.h"

#include "CSSPropertyNames.h"
#include "CSSValueKeywords.h"
#include "DOMPoint.h"
#include "FrameSelection.h"
#include "LegacyRenderSVGResource.h"
#include "LocalFrame.h"
#include "RenderObject.h"
#include "RenderSVGText.h"
#include "SVGElementTypeHelpers.h"
#include "SVGNames.h"
#include "SVGPoint.h"
#include "SVGRect.h"
#include "SVGTextQuery.h"
#include "XMLNames.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGTextContentElement);
 
SVGTextContentElement::SVGTextContentElement(const QualifiedName& tagName, Document& document, UniqueRef<SVGPropertyRegistry>&& propertyRegistry)
    : SVGGraphicsElement(tagName, document, WTFMove(propertyRegistry))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::textLengthAttr, &SVGTextContentElement::m_textLength>();
        PropertyRegistry::registerProperty<SVGNames::lengthAdjustAttr, SVGLengthAdjustType, &SVGTextContentElement::m_lengthAdjust>();
    });
}

unsigned SVGTextContentElement::getNumberOfChars()
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);
    return SVGTextQuery(checkedRenderer().get()).numberOfCharacters();
}

float SVGTextContentElement::getComputedTextLength()
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);
    return SVGTextQuery(checkedRenderer().get()).textLength();
}

ExceptionOr<float> SVGTextContentElement::getSubStringLength(unsigned charnum, unsigned nchars)
{
    unsigned numberOfChars = getNumberOfChars();
    if (charnum >= numberOfChars)
        return Exception { ExceptionCode::IndexSizeError };

    nchars = std::min(nchars, numberOfChars - charnum);
    return SVGTextQuery(checkedRenderer().get()).subStringLength(charnum, nchars);
}

ExceptionOr<Ref<SVGPoint>> SVGTextContentElement::getStartPositionOfChar(unsigned charnum)
{
    if (charnum >= getNumberOfChars())
        return Exception { ExceptionCode::IndexSizeError };

    return SVGPoint::create(SVGTextQuery(checkedRenderer().get()).startPositionOfCharacter(charnum));
}

ExceptionOr<Ref<SVGPoint>> SVGTextContentElement::getEndPositionOfChar(unsigned charnum)
{
    if (charnum >= getNumberOfChars())
        return Exception { ExceptionCode::IndexSizeError };

    return SVGPoint::create(SVGTextQuery(checkedRenderer().get()).endPositionOfCharacter(charnum));
}

ExceptionOr<Ref<SVGRect>> SVGTextContentElement::getExtentOfChar(unsigned charnum)
{
    if (charnum >= getNumberOfChars())
        return Exception { ExceptionCode::IndexSizeError };

    return SVGRect::create(SVGTextQuery(checkedRenderer().get()).extentOfCharacter(charnum));
}

ExceptionOr<float> SVGTextContentElement::getRotationOfChar(unsigned charnum)
{
    if (charnum >= getNumberOfChars())
        return Exception { ExceptionCode::IndexSizeError };

    return SVGTextQuery(checkedRenderer().get()).rotationOfCharacter(charnum);
}

int SVGTextContentElement::getCharNumAtPosition(DOMPointInit&& pointInit)
{
    protectedDocument()->updateLayoutIgnorePendingStylesheets({ LayoutOptions::ContentVisibilityForceLayout }, this);
    FloatPoint transformPoint {static_cast<float>(pointInit.x), static_cast<float>(pointInit.y)};
    return SVGTextQuery(checkedRenderer().get()).characterNumberAtPosition(transformPoint);
}

ExceptionOr<void> SVGTextContentElement::selectSubString(unsigned charnum, unsigned nchars)
{
    unsigned numberOfChars = getNumberOfChars();
    if (charnum >= numberOfChars)
        return Exception { ExceptionCode::IndexSizeError };

    nchars = std::min(nchars, numberOfChars - charnum);

    RefPtr frame = document().frame();
    ASSERT(frame);
    CheckedRef selection = frame->selection();

    // Find selection start
    VisiblePosition start(firstPositionInNode(const_cast<SVGTextContentElement*>(this)));
    for (unsigned i = 0; i < charnum; ++i)
        start = start.next();

    // Find selection end
    VisiblePosition end(start);
    for (unsigned i = 0; i < nchars; ++i)
        end = end.next();

    selection->setSelection(VisibleSelection(start, end));

    return { };
}

bool SVGTextContentElement::hasPresentationalHintsForAttribute(const QualifiedName& name) const
{
    if (name.matches(XMLNames::spaceAttr))
        return true;
    return SVGGraphicsElement::hasPresentationalHintsForAttribute(name);
}

void SVGTextContentElement::collectPresentationalHintsForAttribute(const QualifiedName& name, const AtomString& value, MutableStyleProperties& style)
{
    if (name.matches(XMLNames::spaceAttr)) {
        if (value == "preserve"_s)
            addPropertyToPresentationalHintStyle(style, CSSPropertyWhiteSpaceCollapse, CSSValuePreserve);
        else
            addPropertyToPresentationalHintStyle(style, CSSPropertyWhiteSpaceCollapse, CSSValueCollapse);
        addPropertyToPresentationalHintStyle(style, CSSPropertyTextWrapMode, CSSValueNowrap);
        return;
    }

    SVGGraphicsElement::collectPresentationalHintsForAttribute(name, value, style);
}

void SVGTextContentElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGParsingError parseError = NoError;

    if (name == SVGNames::lengthAdjustAttr) {
        auto propertyValue = SVGPropertyTraits<SVGLengthAdjustType>::fromString(newValue);
        if (propertyValue > 0)
            m_lengthAdjust->setBaseValInternal<SVGLengthAdjustType>(propertyValue);
    } else if (name == SVGNames::textLengthAttr)
        m_textLength->setBaseValInternal(SVGLengthValue::construct(SVGLengthMode::Other, newValue, parseError, SVGLengthNegativeValuesMode::Forbid));

    reportAttributeParsingError(parseError, name, newValue);

    SVGGraphicsElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGTextContentElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        if (attrName == SVGNames::textLengthAttr)
            m_specifiedTextLength = m_textLength->baseVal()->value();

        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        invalidateResourceImageBuffersIfNeeded();
        return;
    }

    SVGGraphicsElement::svgAttributeChanged(attrName);
}

SVGAnimatedLength& SVGTextContentElement::textLengthAnimated()
{
    static NeverDestroyed<SVGLengthValue> defaultTextLength(SVGLengthMode::Other);
    if (m_textLength->baseVal()->value() == defaultTextLength)
        m_textLength->baseVal()->value() = { getComputedTextLength(), SVGLengthType::Number };
    return m_textLength;
}

bool SVGTextContentElement::selfHasRelativeLengths() const
{
    // Any element of the <text> subtree is advertized as using relative lengths.
    // On any window size change, we have to relayout the text subtree, as the
    // effective 'on-screen' font size may change.
    return true;
}

const SVGTextContentElement* SVGTextContentElement::elementFromRenderer(const RenderObject* renderer)
{
    if (!renderer)
        return nullptr;

    if (!renderer->isRenderSVGText() && !renderer->isRenderSVGInline())
        return nullptr;

    auto* element = downcast<SVGElement>(renderer->node());
    ASSERT(element);
    return dynamicDowncast<SVGTextContentElement>(element);
}

}
