/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
#include "RenderSVGGradientStop.h"

#include "ElementInlines.h"
#include "LegacyRenderSVGResourceContainer.h"
#include "RenderSVGGradientStopInlines.h"
#include "RenderSVGResourceGradient.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGradientElement.h"
#include "SVGNames.h"
#include "SVGStopElement.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
    
using namespace SVGNames;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGGradientStop);

RenderSVGGradientStop::RenderSVGGradientStop(SVGStopElement& element, RenderStyle&& style)
    : RenderElement(Type::SVGGradientStop, element, WTFMove(style), { }, { })
{
    ASSERT(isRenderSVGGradientStop());
}

RenderSVGGradientStop::~RenderSVGGradientStop() = default;

void RenderSVGGradientStop::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderElement::styleDidChange(diff, oldStyle);
    if (diff == StyleDifference::Equal)
        return;

    // <stop> elements should only be allowed to make renderers under gradient elements
    // but I can imagine a few cases we might not be catching, so let's not crash if our parent isn't a gradient.
    RefPtr gradient = gradientElement();
    if (!gradient)
        return;

    CheckedPtr renderer = gradient->renderer();
    if (!renderer)
        return;

    if (auto* gradientRenderer = dynamicDowncast<RenderSVGResourceGradient>(renderer.get())) {
        gradientRenderer->invalidateGradient();
        return;
    }

    downcast<LegacyRenderSVGResourceContainer>(*renderer).removeAllClientsFromCacheAndMarkForInvalidation();
}

void RenderSVGGradientStop::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    clearNeedsLayout();
}

SVGGradientElement* RenderSVGGradientStop::gradientElement()
{
    return dynamicDowncast<SVGGradientElement>(element().protectedParentElement().get());
}

}
