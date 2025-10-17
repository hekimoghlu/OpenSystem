/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include "SVGFEMergeNodeElement.h"

#include "LegacyRenderSVGResource.h"
#include "SVGFilterElement.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEMergeNodeElement);

inline SVGFEMergeNodeElement::SVGFEMergeNodeElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feMergeNodeTag));
    
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::inAttr, &SVGFEMergeNodeElement::m_in1>();
    });
}

Ref<SVGFEMergeNodeElement> SVGFEMergeNodeElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEMergeNodeElement(tagName, document));
}

void SVGFEMergeNodeElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::inAttr)
        Ref { m_in1 }->setBaseValInternal(newValue);

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFEMergeNodeElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::inAttr);
        InstanceInvalidationGuard guard(*this);
        SVGFilterPrimitiveStandardAttributes::invalidateFilterPrimitiveParent(this);
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

}
