/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#include "SVGGraphicsElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct DOMPointInit;

enum SVGLengthAdjustType {
    SVGLengthAdjustUnknown,
    SVGLengthAdjustSpacing,
    SVGLengthAdjustSpacingAndGlyphs
};

template<> struct SVGPropertyTraits<SVGLengthAdjustType> {
    static unsigned highestEnumValue() { return SVGLengthAdjustSpacingAndGlyphs; }

    static String toString(SVGLengthAdjustType type)
    {
        switch (type) {
        case SVGLengthAdjustUnknown:
            return emptyString();
        case SVGLengthAdjustSpacing:
            return "spacing"_s;
        case SVGLengthAdjustSpacingAndGlyphs:
            return "spacingAndGlyphs"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static SVGLengthAdjustType fromString(const String& value)
    {
        if (value == "spacingAndGlyphs"_s)
            return SVGLengthAdjustSpacingAndGlyphs;
        if (value == "spacing"_s)
            return SVGLengthAdjustSpacing;
        return SVGLengthAdjustUnknown;
    }
};

class SVGTextContentElement : public SVGGraphicsElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGTextContentElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGTextContentElement);
public:
    enum {
        LENGTHADJUST_UNKNOWN = SVGLengthAdjustUnknown,
        LENGTHADJUST_SPACING = SVGLengthAdjustSpacing,
        LENGTHADJUST_SPACINGANDGLYPHS = SVGLengthAdjustSpacingAndGlyphs
    };

    unsigned getNumberOfChars();
    float getComputedTextLength();
    ExceptionOr<float> getSubStringLength(unsigned charnum, unsigned nchars);
    ExceptionOr<Ref<SVGPoint>> getStartPositionOfChar(unsigned charnum);
    ExceptionOr<Ref<SVGPoint>> getEndPositionOfChar(unsigned charnum);
    ExceptionOr<Ref<SVGRect>> getExtentOfChar(unsigned charnum);
    ExceptionOr<float> getRotationOfChar(unsigned charnum);
    int getCharNumAtPosition(DOMPointInit&&);
    ExceptionOr<void> selectSubString(unsigned charnum, unsigned nchars);

    static const SVGTextContentElement* elementFromRenderer(const RenderObject*);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGTextContentElement, SVGGraphicsElement>;

    const SVGLengthValue& specifiedTextLength() const { return m_specifiedTextLength; }
    const SVGLengthValue& textLength() const { return m_textLength->currentValue(); }
    SVGLengthAdjustType lengthAdjust() const { return m_lengthAdjust->currentValue<SVGLengthAdjustType>(); }

    SVGAnimatedLength& textLengthAnimated();
    SVGAnimatedEnumeration& lengthAdjustAnimated() { return m_lengthAdjust; }

protected:
    SVGTextContentElement(const QualifiedName&, Document&, UniqueRef<SVGPropertyRegistry>&&);

    bool isValid() const override { return SVGTests::isValid(); }

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const override;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool selfHasRelativeLengths() const override;

private:
    bool isTextContent() const final { return true; }

    Ref<SVGAnimatedLength> m_textLength { SVGAnimatedLength::create(this, SVGLengthMode::Other) };
    Ref<SVGAnimatedEnumeration> m_lengthAdjust { SVGAnimatedEnumeration::create(this, SVGLengthAdjustSpacing) };
    SVGLengthValue m_specifiedTextLength { SVGLengthMode::Other };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGTextContentElement)
    static bool isType(const WebCore::SVGElement& element) { return element.isTextContent(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* svgElement = dynamicDowncast<WebCore::SVGElement>(node);
        return svgElement && isType(*svgElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
