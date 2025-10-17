/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#include "RenderSVGResourceRadialGradient.h"

#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceRadialGradientInlines.h"
#include "RenderSVGShape.h"
#include "SVGElementTypeHelpers.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceRadialGradient);

RenderSVGResourceRadialGradient::RenderSVGResourceRadialGradient(SVGRadialGradientElement& element, RenderStyle&& style)
    : RenderSVGResourceGradient(Type::SVGResourceRadialGradient, element, WTFMove(style))
{
}

RenderSVGResourceRadialGradient::~RenderSVGResourceRadialGradient() = default;

void RenderSVGResourceRadialGradient::collectGradientAttributesIfNeeded()
{
    if (m_attributes.has_value())
        return;

    Ref radialGradientElement = this->radialGradientElement();
    radialGradientElement->synchronizeAllAttributes();

    auto attributes = RadialGradientAttributes { };
    if (radialGradientElement->collectGradientAttributes(attributes))
        m_attributes = WTFMove(attributes);
}

RefPtr<Gradient> RenderSVGResourceRadialGradient::createGradient(const RenderStyle& style)
{
    if (!m_attributes)
        return nullptr;

    Ref radialGradientElement = this->radialGradientElement();
    auto centerPoint = SVGLengthContext::resolvePoint(radialGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->cx(), m_attributes->cy());
    auto radius = SVGLengthContext::resolveLength(radialGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->r());

    auto focalPoint = SVGLengthContext::resolvePoint(radialGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->fx(), m_attributes->fy());
    auto focalRadius = SVGLengthContext::resolveLength(radialGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->fr());

    return Gradient::create(
        Gradient::RadialData { focalPoint, centerPoint, focalRadius, radius, 1 },
        { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied },
        platformSpreadMethodFromSVGType(m_attributes->spreadMethod()),
        stopsByApplyingColorFilter(m_attributes->stops(), style),
        RenderingResourceIdentifier::generate()
    );
}

}
