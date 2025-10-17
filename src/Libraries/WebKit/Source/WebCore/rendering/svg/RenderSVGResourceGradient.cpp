/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include "RenderSVGResourceGradient.h"

#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceGradientInlines.h"
#include "RenderSVGShape.h"
#include "SVGRenderStyle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceGradient);

RenderSVGResourceGradient::RenderSVGResourceGradient(Type type, SVGElement& element, RenderStyle&& style)
    : RenderSVGResourcePaintServer(type, element, WTFMove(style))
{
}

RenderSVGResourceGradient::~RenderSVGResourceGradient() = default;

GradientColorStops RenderSVGResourceGradient::stopsByApplyingColorFilter(const GradientColorStops& stops, const RenderStyle& style) const
{
    if (!style.hasAppleColorFilter())
        return stops;

    return stops.mapColors([&] (auto& color) {
        return style.colorByApplyingColorFilter(color);
    });
}

GradientSpreadMethod RenderSVGResourceGradient::platformSpreadMethodFromSVGType(SVGSpreadMethodType method) const
{
    switch (method) {
    case SVGSpreadMethodUnknown:
    case SVGSpreadMethodPad:
        return GradientSpreadMethod::Pad;
    case SVGSpreadMethodReflect:
        return GradientSpreadMethod::Reflect;
    case SVGSpreadMethodRepeat:
        return GradientSpreadMethod::Repeat;
    }

    ASSERT_NOT_REACHED();
    return GradientSpreadMethod::Pad;
}

bool RenderSVGResourceGradient::buildGradientIfNeeded(const RenderLayerModelObject& targetRenderer, const RenderStyle& style, AffineTransform& userspaceTransform)
{
    if (!m_gradient) {
        collectGradientAttributesIfNeeded();
        m_gradient = createGradient(style);

        if (!m_gradient)
            return false;
    }

    auto objectBoundingBox = targetRenderer.objectBoundingBox();
    if (gradientUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) {
        // Gradient is not applicable on 1d objects (empty objectBoundingBox), unless 'gradientUnits' is equal to 'userSpaceOnUse'.
        if (objectBoundingBox.isEmpty())
            return false;

        userspaceTransform.translate(objectBoundingBox.location());
        userspaceTransform.scale(objectBoundingBox.size());
    }

    if (auto gradientTransform = this->gradientTransform(); !gradientTransform.isIdentity())
        userspaceTransform.multiply(gradientTransform);

    return true;
}

bool RenderSVGResourceGradient::prepareFillOperation(GraphicsContext& context, const RenderLayerModelObject& targetRenderer, const RenderStyle& style)
{
    AffineTransform userspaceTransform;
    if (!buildGradientIfNeeded(targetRenderer, style, userspaceTransform))
        return false;

    Ref svgStyle = style.svgStyle();
    context.setAlpha(svgStyle->fillOpacity());
    context.setFillRule(svgStyle->fillRule());
    context.setFillGradient(m_gradient.copyRef().releaseNonNull(), userspaceTransform);
    return true;
}

bool RenderSVGResourceGradient::prepareStrokeOperation(GraphicsContext& context, const RenderLayerModelObject& targetRenderer, const RenderStyle& style)
{
    AffineTransform userspaceTransform;
    if (!buildGradientIfNeeded(targetRenderer, style, userspaceTransform))
        return false;

    Ref svgStyle = style.svgStyle();
    if (svgStyle->vectorEffect() == VectorEffect::NonScalingStroke) {
        if (CheckedPtr shape = dynamicDowncast<RenderSVGShape>(targetRenderer))
            userspaceTransform = shape->nonScalingStrokeTransform() * userspaceTransform;
    }

    context.setAlpha(svgStyle->strokeOpacity());
    SVGRenderSupport::applyStrokeStyleToContext(context, style, targetRenderer);
    context.setStrokeGradient(m_gradient.copyRef().releaseNonNull(), userspaceTransform);
    return true;
}

}
