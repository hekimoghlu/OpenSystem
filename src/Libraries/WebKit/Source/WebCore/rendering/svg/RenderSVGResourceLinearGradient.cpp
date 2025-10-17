/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#include "RenderSVGResourceLinearGradient.h"

#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceLinearGradientInlines.h"
#include "SVGElementTypeHelpers.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceLinearGradient);

RenderSVGResourceLinearGradient::RenderSVGResourceLinearGradient(SVGLinearGradientElement& element, RenderStyle&& style)
    : RenderSVGResourceGradient(Type::SVGResourceLinearGradient, element, WTFMove(style))
{
}

RenderSVGResourceLinearGradient::~RenderSVGResourceLinearGradient() = default;

void RenderSVGResourceLinearGradient::collectGradientAttributesIfNeeded()
{
    if (m_attributes.has_value())
        return;

    Ref linearGradientElement = this->linearGradientElement();
    linearGradientElement->synchronizeAllAttributes();

    auto attributes = LinearGradientAttributes { };
    if (linearGradientElement->collectGradientAttributes(attributes))
        m_attributes = WTFMove(attributes);
}

RefPtr<Gradient> RenderSVGResourceLinearGradient::createGradient(const RenderStyle& style)
{
    if (!m_attributes)
        return nullptr;

    Ref linearGradientElement = this->linearGradientElement();
    auto startPoint = SVGLengthContext::resolvePoint(linearGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->x1(), m_attributes->y1());
    auto endPoint = SVGLengthContext::resolvePoint(linearGradientElement.ptr(), m_attributes->gradientUnits(), m_attributes->x2(), m_attributes->y2());

    return Gradient::create(
        Gradient::LinearData { startPoint, endPoint },
        { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied },
        platformSpreadMethodFromSVGType(m_attributes->spreadMethod()),
        stopsByApplyingColorFilter(m_attributes->stops(), style),
        RenderingResourceIdentifier::generate()
    );
}

}

