/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#include "Gradient.h"

#if USE(SKIA)

#include "AffineTransform.h"
#include "GradientColorStops.h"
#include "GraphicsContextSkia.h"
#include "NotImplemented.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkColor.h>
#include <skia/core/SkScalar.h>
#include <skia/effects/SkGradientShader.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

void Gradient::stopsChanged()
{
    m_shader = { };
}

inline SkScalar webCoreDoubleToSkScalar(double d)
{
    return SkDoubleToScalar(std::isfinite(d) ? d : 0);
}

static SkGradientShader::Interpolation toSkiaInterpolation(const ColorInterpolationMethod& method)
{
    SkGradientShader::Interpolation interpolation;

    WTF::switchOn(method.colorSpace,
        [&] (const ColorInterpolationMethod::HSL&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kHSL;
        },
        [&] (const ColorInterpolationMethod::HWB&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kHWB;
        },
        [&] (const ColorInterpolationMethod::LCH&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kLCH;
        },
        [&] (const ColorInterpolationMethod::Lab&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kLab;
        },
        [&] (const ColorInterpolationMethod::OKLCH&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kOKLCH;
        },
        [&] (const ColorInterpolationMethod::OKLab&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kOKLab;
        },
        [&] (const ColorInterpolationMethod::SRGB&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kSRGB;
        },
        [&] (const ColorInterpolationMethod::SRGBLinear&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kSRGBLinear;
        },
        [&] (const ColorInterpolationMethod::XYZD50&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kSRGBLinear;
        },
        [&] (const ColorInterpolationMethod::XYZD65&) {
            interpolation.fColorSpace = SkGradientShader::Interpolation::ColorSpace::kSRGBLinear;
        },
        [&] (const auto&) {
            // FIXME: Support other color spaces once skia has support for them.
        });

    switch (method.alphaPremultiplication) {
    case AlphaPremultiplication::Premultiplied:
        interpolation.fInPremul = SkGradientShader::Interpolation::InPremul::kYes;
        break;
    case AlphaPremultiplication::Unpremultiplied:
        interpolation.fInPremul = SkGradientShader::Interpolation::InPremul::kNo;
        break;
    }

    return interpolation;
}

sk_sp<SkShader> Gradient::shader(float globalAlpha, const AffineTransform& gradientSpaceTransform)
{
    if (m_shader)
        return m_shader;

    Vector<SkColor4f, 8> colors;
    colors.reserveInitialCapacity(stops().size());
    Vector<SkScalar, 8> positions;
    positions.reserveInitialCapacity(stops().size());
    auto fillStops = [&](const GradientColorStops::StopVector& stops) {
        if (stops.isEmpty()) {
            positions.append(webCoreDoubleToSkScalar(0));
            colors.append(SkColors::kTransparent);
        } else if (stops.begin()->offset > 0) {
            positions.append(webCoreDoubleToSkScalar(0));
            colors.append(stops.begin()->color.colorWithAlphaMultipliedBy(globalAlpha));
        }

        for (size_t i = 0; i < stops.size(); i++) {
            positions.append(webCoreDoubleToSkScalar(stops[i].offset));
            colors.append(stops[i].color.colorWithAlphaMultipliedBy(globalAlpha));
        }

        if (positions.last() < 1) {
            positions.append(webCoreDoubleToSkScalar(1));
            colors.append(colors.last());
        }
    };
    fillStops(stops().sorted().stops());

    SkTileMode tileMode = SkTileMode::kClamp;
    switch (m_spreadMethod) {
    case GradientSpreadMethod::Pad:
        tileMode = SkTileMode::kClamp;
        break;
    case GradientSpreadMethod::Reflect:
        tileMode = SkTileMode::kMirror;
        break;
    case GradientSpreadMethod::Repeat:
        tileMode = SkTileMode::kRepeat;
        break;
    }

    auto interpolation = toSkiaInterpolation(colorInterpolationMethod());
    SkMatrix matrix = gradientSpaceTransform;

    m_shader = WTF::switchOn(
        m_data,
        [&](const LinearData& data) {
            SkPoint points[] = { SkPoint::Make(data.point0.x(), data.point0.y()), SkPoint::Make(data.point1.x(), data.point1.y()) };

            return SkGradientShader::MakeLinear(points, colors.data(), nullptr, positions.data(), colors.size(), tileMode, interpolation, &matrix);
        },
        [&](const RadialData& data) {
            if (data.aspectRatio != 1)
                matrix.preScale(1, 1 / data.aspectRatio, data.point0.x(), data.point0.y());

            SkPoint start = SkPoint::Make(data.point0.x(), data.point0.y());
            SkPoint end = SkPoint::Make(data.point1.x(), data.point1.y());
            SkScalar startRadius = std::max(webCoreDoubleToSkScalar(data.startRadius), 0.0f);
            SkScalar endRadius = std::max(webCoreDoubleToSkScalar(data.endRadius), 0.0f);

            return SkGradientShader::MakeTwoPointConical(start, startRadius, end, endRadius, colors.data(), nullptr, positions.data(), colors.size(), tileMode, interpolation, &matrix);
        },
        [&](const ConicData& data) {
            // Skia's renders it tilted by 90 degrees, so offset that rotation in the matrix
            matrix.preRotate(SkRadiansToDegrees(data.angleRadians) - 90.0f, data.point0.x(), data.point0.y());

            return SkGradientShader::MakeSweep(data.point0.x(), data.point0.y(), colors.data(), nullptr, positions.data(), colors.size(), tileMode, 0, 360, interpolation, &matrix);
        });

    return m_shader;
}

void Gradient::fill(GraphicsContext& context, const FloatRect& rect)
{
    auto paint = static_cast<GraphicsContextSkia*>(&context)->createFillPaint();
    paint.setShader(shader(context.alpha(), context.fillGradientSpaceTransform()));
    context.platformContext()->drawRect(rect, paint);
}

} // namespace WebCore

#endif // USE(SKIA)
