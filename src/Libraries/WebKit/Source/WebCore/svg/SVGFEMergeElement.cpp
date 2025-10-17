/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include "SVGFEMergeElement.h"

#include "ElementChildIteratorInlines.h"
#include "FEMerge.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFEMergeNodeElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEMergeElement);

inline SVGFEMergeElement::SVGFEMergeElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(SVGNames::feMergeTag));
}

Ref<SVGFEMergeElement> SVGFEMergeElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEMergeElement(tagName, document));
}

void SVGFEMergeElement::childrenChanged(const ChildChange& change)
{
    SVGFilterPrimitiveStandardAttributes::childrenChanged(change);
    InstanceInvalidationGuard guard(*this);
    markFilterEffectForRebuild();
}

Vector<AtomString> SVGFEMergeElement::filterEffectInputsNames() const
{
    Vector<AtomString> inputsNames;
    for (auto& mergeNode : childrenOfType<SVGFEMergeNodeElement>(*this))
        inputsNames.append(mergeNode.in1());
    return inputsNames;
}

RefPtr<FilterEffect> SVGFEMergeElement::createFilterEffect(const FilterEffectVector& inputs, const GraphicsContext&) const
{
    return FEMerge::create(inputs.size());
}

} // namespace WebCore
