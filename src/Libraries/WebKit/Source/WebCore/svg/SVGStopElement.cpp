/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "SVGStopElement.h"

#include "Document.h"
#include "LegacyRenderSVGResource.h"
#include "RenderSVGGradientStop.h"
#include "SVGGradientElement.h"
#include "SVGNames.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGStopElement);

inline SVGStopElement::SVGStopElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::stopTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::offsetAttr, &SVGStopElement::m_offset>();
    });
}

Ref<SVGStopElement> SVGStopElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGStopElement(tagName, document));
}

void SVGStopElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::offsetAttr) {
        if (newValue.endsWith('%'))
            Ref { m_offset }->setBaseValInternal(newValue.string().left(newValue.length() - 1).toFloat() / 100.0f);
        else
            Ref { m_offset }->setBaseValInternal(newValue.toFloat());
    }

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGStopElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::offsetAttr);
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

RenderPtr<RenderElement> SVGStopElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderSVGGradientStop>(*this, WTFMove(style));
}

bool SVGStopElement::rendererIsNeeded(const RenderStyle&)
{
    return true;
}

Color SVGStopElement::stopColorIncludingOpacity() const
{
    // Return initial value 'black' as per web specification below:
    // https://svgwg.org/svg2-draft/pservers.html#StopColorProperties
    if (!renderer())
        return Color::black;

    auto& style = renderer()->style();
    Ref svgStyle = style.svgStyle();
    auto stopColor = style.colorResolvingCurrentColor(svgStyle->stopColor());

    return stopColor.colorWithAlphaMultipliedBy(svgStyle->stopOpacity());
}

}
