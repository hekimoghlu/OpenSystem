/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#include "RenderSVGTransformableContainer.h"

#include "RenderSVGModelObjectInlines.h"
#include "SVGContainerLayout.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGElement.h"
#include "SVGGraphicsElement.h"
#include "SVGUseElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGTransformableContainer);

RenderSVGTransformableContainer::RenderSVGTransformableContainer(SVGGraphicsElement& element, RenderStyle&& style)
    : RenderSVGContainer(Type::SVGTransformableContainer, element, WTFMove(style))
{
    ASSERT(isRenderSVGTransformableContainer());
}

RenderSVGTransformableContainer::~RenderSVGTransformableContainer() = default;

SVGGraphicsElement& RenderSVGTransformableContainer::graphicsElement() const
{
    return downcast<SVGGraphicsElement>(RenderSVGContainer::element());
}

Ref<SVGGraphicsElement> RenderSVGTransformableContainer::protectedGraphicsElement() const
{
    return graphicsElement();
}

inline SVGUseElement* associatedUseElement(SVGGraphicsElement& element)
{
    // If we're either the renderer for a <use> element, or for any <g> element inside the shadow
    // tree, that was created during the use/symbol/svg expansion in SVGUseElement. These containers
    // need to respect the translations induced by their corresponding use elements x/y attributes.
    if (auto* useElement = dynamicDowncast<SVGUseElement>(element))
        return useElement;

    if (element.isInShadowTree() && is<SVGGElement>(element)) {
        if (auto* useElement = dynamicDowncast<SVGUseElement>(element.correspondingElement()))
            return useElement;
    }

    return nullptr;
}

FloatSize RenderSVGTransformableContainer::additionalContainerTranslation() const
{
    Ref graphicsElement = this->graphicsElement();
    if (RefPtr useElement = associatedUseElement(graphicsElement)) {
        SVGLengthContext lengthContext(graphicsElement.ptr());
        return { useElement->x().value(lengthContext), useElement->y().value(lengthContext) };
    }

    return { };
}

bool RenderSVGTransformableContainer::needsHasSVGTransformFlags() const
{
    Ref graphicsElement = this->graphicsElement();
    return graphicsElement->hasTransformRelatedAttributes() || associatedUseElement(graphicsElement);
}

void RenderSVGTransformableContainer::updateLayerTransform()
{
    // First update the supplemental layer transform...
    m_supplementalLayerTransform = AffineTransform::makeTranslation(additionalContainerTranslation());

    // ... before being able to use it in RenderLayerModelObject::updateLayerTransform().
    RenderSVGContainer::updateLayerTransform();
}

void RenderSVGTransformableContainer::applyTransform(TransformationMatrix& transform, const RenderStyle& style, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption> options) const
{
    auto postTransform = m_supplementalLayerTransform.isIdentity() ? std::nullopt : std::make_optional(m_supplementalLayerTransform);
    applySVGTransform(transform, protectedGraphicsElement(), style, boundingBox, std::nullopt, postTransform, options);
}

}
