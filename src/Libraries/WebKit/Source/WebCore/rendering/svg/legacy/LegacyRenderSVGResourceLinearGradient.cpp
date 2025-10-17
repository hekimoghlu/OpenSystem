/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#include "LegacyRenderSVGResourceLinearGradient.h"

#include "LegacyRenderSVGResourceLinearGradientInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGResourceLinearGradient);

LegacyRenderSVGResourceLinearGradient::LegacyRenderSVGResourceLinearGradient(SVGLinearGradientElement& element, RenderStyle&& style)
    : LegacyRenderSVGResourceGradient(Type::LegacySVGResourceLinearGradient, element, WTFMove(style))
{
}

LegacyRenderSVGResourceLinearGradient::~LegacyRenderSVGResourceLinearGradient() = default;

bool LegacyRenderSVGResourceLinearGradient::collectGradientAttributes()
{
    m_attributes = LinearGradientAttributes();
    return protectedLinearGradientElement()->collectGradientAttributes(m_attributes);
}

FloatPoint LegacyRenderSVGResourceLinearGradient::startPoint(const LinearGradientAttributes& attributes) const
{
    return SVGLengthContext::resolvePoint(protectedLinearGradientElement().ptr(), attributes.gradientUnits(), attributes.x1(), attributes.y1());
}

FloatPoint LegacyRenderSVGResourceLinearGradient::endPoint(const LinearGradientAttributes& attributes) const
{
    return SVGLengthContext::resolvePoint(protectedLinearGradientElement().ptr(), attributes.gradientUnits(), attributes.x2(), attributes.y2());
}

Ref<Gradient> LegacyRenderSVGResourceLinearGradient::buildGradient(const RenderStyle& style) const
{
    return Gradient::create(
        Gradient::LinearData { startPoint(m_attributes), endPoint(m_attributes) },
        { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied },
        platformSpreadMethodFromSVGType(m_attributes.spreadMethod()),
        stopsByApplyingColorFilter(m_attributes.stops(), style),
        RenderingResourceIdentifier::generate()
    );
}

}
