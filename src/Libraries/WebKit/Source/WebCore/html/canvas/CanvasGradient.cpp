/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "CanvasGradient.h"

#include "CanvasStyle.h"
#include "Gradient.h"

namespace WebCore {

CanvasGradient::CanvasGradient(const FloatPoint& p0, const FloatPoint& p1)
    : m_gradient(Gradient::create(Gradient::LinearData { p0, p1 }, { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied }))
{
}

CanvasGradient::CanvasGradient(const FloatPoint& p0, float r0, const FloatPoint& p1, float r1)
    : m_gradient(Gradient::create(Gradient::RadialData { p0, p1, r0, r1, 1 }, { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied }))
{
}

CanvasGradient::CanvasGradient(const FloatPoint& centerPoint, float angleInRadians)
    : m_gradient(Gradient::create(Gradient::ConicData { centerPoint, angleInRadians }, { ColorInterpolationMethod::SRGB { }, AlphaPremultiplication::Unpremultiplied }))
{
}

Ref<CanvasGradient> CanvasGradient::create(const FloatPoint& p0, const FloatPoint& p1)
{
    return adoptRef(*new CanvasGradient(p0, p1));
}

Ref<CanvasGradient> CanvasGradient::create(const FloatPoint& p0, float r0, const FloatPoint& p1, float r1)
{
    return adoptRef(*new CanvasGradient(p0, r0, p1, r1));
}

Ref<CanvasGradient> CanvasGradient::create(const FloatPoint& centerPoint, float angleInRadians)
{
    return adoptRef(*new CanvasGradient(centerPoint, angleInRadians));
}

CanvasGradient::~CanvasGradient() = default;

ExceptionOr<void> CanvasGradient::addColorStop(ScriptExecutionContext& scriptExecutionContext, double value, const String& colorString)
{
    if (!(value >= 0 && value <= 1))
        return Exception { ExceptionCode::IndexSizeError };

    auto color = parseColor(colorString, scriptExecutionContext);
    if (!color.isValid())
        return Exception { ExceptionCode::SyntaxError };

    m_gradient->addColorStop({ static_cast<float>(value), WTFMove(color) });
    return { };
}

} // namespace WebCore
