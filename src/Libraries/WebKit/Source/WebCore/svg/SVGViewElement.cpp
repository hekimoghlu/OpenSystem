/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "SVGViewElement.h"

#include "LegacyRenderSVGResource.h"
#include "RenderElement.h"
#include "SVGNames.h"
#include "SVGSVGElement.h"
#include "SVGStringList.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGViewElement);

inline SVGViewElement::SVGViewElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGFitToViewBox(this)
{
    ASSERT(hasTagName(SVGNames::viewTag));
}

Ref<SVGViewElement> SVGViewElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGViewElement(tagName, document));
}

void SVGViewElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGFitToViewBox::parseAttribute(name, newValue);
    SVGZoomAndPan::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGViewElement::svgAttributeChanged(const QualifiedName& attrName)
{
    // We ignore changes to SVGNames::viewTargetAttr, which is deprecated and unused in WebCore.
    if (PropertyRegistry::isKnownAttribute(attrName))
        return;

    if (SVGFitToViewBox::isKnownAttribute(attrName)) {
        RefPtr targetElement = m_targetElement.get();
        if (!targetElement)
            return;
        targetElement->inheritViewAttributes(*this);
        targetElement->updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

}
