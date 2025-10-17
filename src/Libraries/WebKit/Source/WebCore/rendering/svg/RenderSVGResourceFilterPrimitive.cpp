/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#include "RenderSVGResourceFilterPrimitive.h"

#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceFilter.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFEDiffuseLightingElement.h"
#include "SVGFEDropShadowElement.h"
#include "SVGFEFloodElement.h"
#include "SVGFESpecularLightingElement.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include "SVGNames.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceFilterPrimitive);

RenderSVGResourceFilterPrimitive::RenderSVGResourceFilterPrimitive(SVGFilterPrimitiveStandardAttributes& element, RenderStyle&& style)
    : RenderSVGHiddenContainer(Type::SVGResourceFilterPrimitive, element, WTFMove(style))
{
    ASSERT(isRenderSVGResourceFilterPrimitive());
}

void RenderSVGResourceFilterPrimitive::markFilterEffectForRepaint(FilterEffect* effect)
{
    if (!effect)
        return;

    CheckedPtr parent = dynamicDowncast<RenderSVGResourceFilter>(this->parent());
    if (!parent)
        return;

    parent->invalidateFilter();
}

void RenderSVGResourceFilterPrimitive::markFilterEffectForRebuild()
{
    CheckedPtr parent = dynamicDowncast<RenderSVGResourceFilter>(this->parent());
    if (!parent)
        return;

    parent->invalidateFilter();
}

SVGFilterPrimitiveStandardAttributes& RenderSVGResourceFilterPrimitive::filterPrimitiveElement() const
{
    return static_cast<SVGFilterPrimitiveStandardAttributes&>(RenderSVGHiddenContainer::element());
}

void RenderSVGResourceFilterPrimitive::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderSVGHiddenContainer::styleDidChange(diff, oldStyle);

    if (diff == StyleDifference::Equal || !oldStyle)
        return;

    Ref newStyle = style().svgStyle();
    if (is<SVGFEFloodElement>(filterPrimitiveElement()) || is<SVGFEDropShadowElement>(filterPrimitiveElement())) {
        if (newStyle->floodColor() != oldStyle->svgStyle().floodColor())
            filterPrimitiveElement().primitiveAttributeChanged(SVGNames::flood_colorAttr);
        if (newStyle->floodOpacity() != oldStyle->svgStyle().floodOpacity())
            filterPrimitiveElement().primitiveAttributeChanged(SVGNames::flood_opacityAttr);
        return;
    }
    if (is<SVGFEDiffuseLightingElement>(filterPrimitiveElement()) || is<SVGFESpecularLightingElement>(filterPrimitiveElement())) {
        if (newStyle->lightingColor() != oldStyle->svgStyle().lightingColor())
            filterPrimitiveElement().primitiveAttributeChanged(SVGNames::lighting_colorAttr);
    }
}

} // namespace WebCore
