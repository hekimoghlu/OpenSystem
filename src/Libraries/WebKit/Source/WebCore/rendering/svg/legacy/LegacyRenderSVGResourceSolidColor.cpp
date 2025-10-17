/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "LegacyRenderSVGResourceSolidColor.h"

#include "GraphicsContext.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "RenderElement.h"
#include "RenderStyle.h"
#include "RenderView.h"
#include "SVGRenderStyle.h"
#include "SVGRenderSupport.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LegacyRenderSVGResourceSolidColor);

LegacyRenderSVGResourceSolidColor::LegacyRenderSVGResourceSolidColor() = default;

LegacyRenderSVGResourceSolidColor::~LegacyRenderSVGResourceSolidColor() = default;

auto LegacyRenderSVGResourceSolidColor::applyResource(RenderElement& renderer, const RenderStyle& style, GraphicsContext*& context, OptionSet<RenderSVGResourceMode> resourceMode) -> OptionSet<ApplyResult>
{
    ASSERT(context);
    ASSERT(!resourceMode.isEmpty());

    const SVGRenderStyle& svgStyle = style.svgStyle();

    bool isRenderingMask = renderer.view().frameView().paintBehavior().contains(PaintBehavior::RenderingSVGClipOrMask);

    if (resourceMode.contains(RenderSVGResourceMode::ApplyToFill)) {
        if (!isRenderingMask)
            context->setAlpha(svgStyle.fillOpacity());
        else
            context->setAlpha(1);
        context->setFillColor(style.colorByApplyingColorFilter(m_color));
        if (isRenderingMask)
            context->setFillRule(svgStyle.clipRule());
        else
            context->setFillRule(svgStyle.fillRule());

        if (resourceMode.contains(RenderSVGResourceMode::ApplyToText))
            context->setTextDrawingMode(TextDrawingMode::Fill);
    } else if (resourceMode.contains(RenderSVGResourceMode::ApplyToStroke)) {
        // When rendering the mask for a LegacyRenderSVGResourceClipper, the stroke code path is never hit.
        ASSERT(!isRenderingMask);
        context->setAlpha(svgStyle.strokeOpacity());
        context->setStrokeColor(style.colorByApplyingColorFilter(m_color));

        SVGRenderSupport::applyStrokeStyleToContext(*context, style, renderer);

        if (resourceMode.contains(RenderSVGResourceMode::ApplyToText))
            context->setTextDrawingMode(TextDrawingMode::Stroke);
    }

    return { ApplyResult::ResourceApplied };
}

void LegacyRenderSVGResourceSolidColor::postApplyResource(RenderElement&, GraphicsContext*& context, OptionSet<RenderSVGResourceMode> resourceMode, const Path* path, const RenderElement* shape)
{
    ASSERT(context);
    ASSERT(!resourceMode.isEmpty());
    fillAndStrokePathOrShape(*context, resourceMode, path, shape);
}

}
