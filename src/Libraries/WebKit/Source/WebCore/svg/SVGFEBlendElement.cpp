/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#include "SVGFEBlendElement.h"

#include "FEBlend.h"
#include "NodeName.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEBlendElement);

inline SVGFEBlendElement::SVGFEBlendElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feBlendTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::modeAttr, BlendMode, &SVGFEBlendElement::m_mode>();
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEBlendElement::m_in1>();
        PropertyRegistry::registerProperty<SVGNames::in2Attr, &SVGFEBlendElement::m_in2>();
    });
}

Ref<SVGFEBlendElement> SVGFEBlendElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEBlendElement(tagName, document));
}

void SVGFEBlendElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::modeAttr: {
        BlendMode mode = BlendMode::Normal;
        if (parseBlendMode(newValue, mode))
        Ref { m_mode }->setBaseValInternal<BlendMode>(mode);
        break;
    }
    case AttributeNames::inAttr:
        Ref { m_in1 }->setBaseValInternal(newValue);
        break;
    case AttributeNames::in2Attr:
        Ref { m_in2 }->setBaseValInternal(newValue);
        break;
    default:
        break;
    }

    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

bool SVGFEBlendElement::setFilterEffectAttribute(FilterEffect& effect, const QualifiedName& attrName)
{
    auto& feBlend = downcast<FEBlend>(effect);
    if (attrName == SVGNames::modeAttr)
        return feBlend.setBlendMode(mode());

    ASSERT_NOT_REACHED();
    return false;
}

void SVGFEBlendElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        if (attrName == SVGNames::modeAttr)
            primitiveAttributeChanged(attrName);
        else {
            ASSERT(attrName == SVGNames::inAttr || attrName == SVGNames::in2Attr);
            updateSVGRendererForElementChange();
        }
        return;
    }

    SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
}

RefPtr<FilterEffect> SVGFEBlendElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext&) const
{
    return FEBlend::create(mode());
}

} // namespace WebCore
