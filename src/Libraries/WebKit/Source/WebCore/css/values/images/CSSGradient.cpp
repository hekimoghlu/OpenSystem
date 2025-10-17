/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "CSSGradient.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "CSSPrimitiveValueMappings.h"
#include "CalculationValue.h"
#include "ColorInterpolation.h"
#include "StyleBuilderConverter.h"
#include "StyleGradientImage.h"
#include "StylePosition.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

// MARK: - Gradient Color Stop

template<typename C, typename P> static void colorStopSerializationForCSS(StringBuilder& builder, const GradientColorStop<C, P>& stop)
{
    if (stop.color && stop.position) {
        serializationForCSS(builder, *stop.color);
        builder.append(' ');
        serializationForCSS(builder, *stop.position);
    } else if (stop.color)
        serializationForCSS(builder, *stop.color);
    else if (stop.position)
        serializationForCSS(builder, *stop.position);
}

void Serialize<GradientAngularColorStop>::operator()(StringBuilder& builder, const GradientAngularColorStop& stop)
{
    colorStopSerializationForCSS(builder, stop);
}

void Serialize<GradientLinearColorStop>::operator()(StringBuilder& builder, const GradientLinearColorStop& stop)
{
    colorStopSerializationForCSS(builder, stop);
}

void Serialize<GradientDeprecatedColorStop>::operator()(StringBuilder& builder, const GradientDeprecatedColorStop& stop)
{
    auto appendRaw = [&](const auto& color, NumberRaw<> raw) {
        if (!raw.value) {
            builder.append("from("_s);
            serializationForCSS(builder, color);
            builder.append(')');
        } else if (raw.value == 1) {
            builder.append("to("_s);
            serializationForCSS(builder, color);
            builder.append(')');
        } else {
            builder.append("color-stop("_s);
            serializationForCSS(builder, raw);
            builder.append(", "_s);
            serializationForCSS(builder, color);
            builder.append(')');
        }
    };

    auto appendCalc = [&](const auto& color, const auto& calc) {
        builder.append("color-stop("_s);
        serializationForCSS(builder, calc);
        builder.append(", "_s);
        serializationForCSS(builder, color);
        builder.append(')');
    };

    WTF::switchOn(stop.position,
        [&](const Number<>& number) {
            return WTF::switchOn(number,
                [&](const Number<>::Raw& raw) {
                    appendRaw(stop.color, raw);
                },
                [&](const Number<>::Calc& calc) {
                    appendCalc(stop.color, calc);
                }
            );
        },
        [&](const Percentage<>& percentage) {
            return WTF::switchOn(percentage,
                [&](const Percentage<>::Raw& raw) {
                    appendRaw(stop.color, { raw.value / 100.0 });
                },
                [&](const Percentage<>::Calc& calc) {
                    appendCalc(stop.color, calc);
                }
            );
        }
    );
}

// MARK: - Gradient Color Interpolation

static bool appendColorInterpolationMethod(StringBuilder& builder, CSS::GradientColorInterpolationMethod colorInterpolationMethod, bool needsLeadingSpace)
{
    return WTF::switchOn(colorInterpolationMethod.method.colorSpace,
        [&](const ColorInterpolationMethod::OKLab&) {
            if (colorInterpolationMethod.defaultMethod != CSS::GradientColorInterpolationMethod::Default::OKLab) {
                builder.append(needsLeadingSpace ? " "_s : ""_s, "in oklab"_s);
                return true;
            }
            return false;
        },
        [&](const ColorInterpolationMethod::SRGB&) {
            if (colorInterpolationMethod.defaultMethod != CSS::GradientColorInterpolationMethod::Default::SRGB) {
                builder.append(needsLeadingSpace ? " "_s : ""_s, "in srgb"_s);
                return true;
            }
            return false;
        },
        [&]<typename MethodColorSpace>(const MethodColorSpace& methodColorSpace) {
            builder.append(needsLeadingSpace ? " "_s : ""_s, "in "_s, serializationForCSS(methodColorSpace.interpolationColorSpace));
            if constexpr (hasHueInterpolationMethod<MethodColorSpace>)
                serializationForCSS(builder, methodColorSpace.hueInterpolationMethod);
            return true;
        }
    );
}

// MARK: - LinearGradient

void Serialize<LinearGradient>::operator()(StringBuilder& builder, const LinearGradient& gradient)
{
    bool wroteSomething = false;

    WTF::switchOn(gradient.gradientLine,
        [&](const Angle<>& angle) {
            WTF::switchOn(angle,
                [&](const Angle<>::Raw& angleRaw) {
                    if (convertToValueInUnitsOf<AngleUnit::Deg>(angleRaw) == 180)
                        return;

                    serializationForCSS(builder, angleRaw);
                    wroteSomething = true;
                },
                [&](const Angle<>::Calc& angleCalc) {
                    serializationForCSS(builder, angleCalc);
                    wroteSomething = true;
                }
            );
        },
        [&](const Horizontal& horizontal) {
            builder.append("to "_s);
            serializationForCSS(builder, horizontal);
            wroteSomething = true;
        },
        [&](const Vertical& vertical) {
            if (std::holds_alternative<Keyword::Bottom>(vertical))
                return;

            builder.append("to "_s);
            serializationForCSS(builder, vertical);
            wroteSomething = true;
        },
        [&](const SpaceSeparatedTuple<Horizontal, Vertical>& pair) {
            builder.append("to "_s);
            serializationForCSS(builder, pair);
            wroteSomething = true;
        }
    );

    if (appendColorInterpolationMethod(builder, gradient.colorInterpolationMethod, wroteSomething))
        wroteSomething = true;

    if (wroteSomething)
        builder.append(", "_s);

    serializationForCSS(builder, gradient.stops);
}

// MARK: - PrefixedLinearGradient

void Serialize<PrefixedLinearGradient>::operator()(StringBuilder& builder, const PrefixedLinearGradient& gradient)
{
    serializationForCSS(builder, gradient.gradientLine);
    builder.append(", "_s);
    serializationForCSS(builder, gradient.stops);
}

// MARK: - DeprecatedLinearGradient

void Serialize<DeprecatedLinearGradient>::operator()(StringBuilder& builder, const DeprecatedLinearGradient& gradient)
{
    builder.append("linear, "_s);

    serializationForCSS(builder, gradient.gradientLine);

    if (!gradient.stops.isEmpty()) {
        builder.append(", "_s);
        serializationForCSS(builder, gradient.stops);
    }
}

// MARK: - RadialGradient

void Serialize<RadialGradient::Ellipse>::operator()(StringBuilder& builder, const RadialGradient::Ellipse& ellipse)
{
    auto lengthBefore = builder.length();

    WTF::switchOn(ellipse.size,
        [&](const RadialGradient::Ellipse::Size& size) {
            serializationForCSS(builder, size);
        },
        [&](const RadialGradient::Extent& extent) {
            if (!std::holds_alternative<Keyword::FarthestCorner>(extent))
                serializationForCSS(builder, extent);
        }
    );

    if (ellipse.position) {
        if (!isCenterPosition(*ellipse.position)) {
            bool wroteSomething = builder.length() != lengthBefore;
            if (wroteSomething)
                builder.append(' ');

            builder.append("at "_s);
            serializationForCSS(builder, *ellipse.position);
        }
    }
}

void Serialize<RadialGradient::Circle>::operator()(StringBuilder& builder, const RadialGradient::Circle& circle)
{
    WTF::switchOn(circle.size,
        [&](const RadialGradient::Circle::Length& length) {
            serializationForCSS(builder, length);
        },
        [&](const RadialGradient::Extent& extent) {
            if (!std::holds_alternative<Keyword::FarthestCorner>(extent)) {
                builder.append("circle "_s);
                serializationForCSS(builder, extent);
            } else
                builder.append("circle"_s);
        }
    );

    if (circle.position) {
        if (!isCenterPosition(*circle.position)) {
            builder.append(" at "_s);
            serializationForCSS(builder, *circle.position);
        }
    }
}

void Serialize<RadialGradient>::operator()(StringBuilder& builder, const RadialGradient& gradient)
{
    auto lengthBefore = builder.length();
    serializationForCSS(builder, gradient.gradientBox);
    bool wroteSomething = builder.length() != lengthBefore;

    if (appendColorInterpolationMethod(builder, gradient.colorInterpolationMethod, wroteSomething))
        wroteSomething = true;

    if (wroteSomething)
        builder.append(", "_s);

    serializationForCSS(builder, gradient.stops);
}

// MARK: - PrefixedRadialGradient

void Serialize<PrefixedRadialGradient::Ellipse>::operator()(StringBuilder& builder, const PrefixedRadialGradient::Ellipse& ellipse)
{
    if (ellipse.position)
        serializationForCSS(builder, *ellipse.position);
    else
        builder.append("center"_s);

    if (ellipse.size) {
        WTF::switchOn(*ellipse.size,
            [&](const PrefixedRadialGradient::Ellipse::Size& size) {
                builder.append(", "_s);
                serializationForCSS(builder, size);
            },
            [&](const PrefixedRadialGradient::Extent& extent) {
                builder.append(", ellipse "_s);
                serializationForCSS(builder, extent);
            }
        );
    }
}

void Serialize<PrefixedRadialGradient::Circle>::operator()(StringBuilder& builder, const PrefixedRadialGradient::Circle& circle)
{
    if (circle.position)
        serializationForCSS(builder, *circle.position);
    else
        builder.append("center"_s);

    builder.append(", circle "_s);
    serializationForCSS(builder, circle.size.value_or(PrefixedRadialGradient::Extent { CSS::Keyword::Cover { } }));
}

void Serialize<PrefixedRadialGradient>::operator()(StringBuilder& builder, const PrefixedRadialGradient& gradient)
{
    auto lengthBefore = builder.length();
    serializationForCSS(builder, gradient.gradientBox);
    bool wroteSomething = builder.length() != lengthBefore;

    if (wroteSomething)
        builder.append(", "_s);

    serializationForCSS(builder, gradient.stops);
}

// MARK: - DeprecatedRadialGradient

void Serialize<DeprecatedRadialGradient::GradientBox>::operator()(StringBuilder& builder, const DeprecatedRadialGradient::GradientBox& gradientBox)
{
    serializationForCSS(builder, gradientBox.first);
    builder.append(", "_s);
    serializationForCSS(builder, gradientBox.firstRadius);
    builder.append(", "_s);
    serializationForCSS(builder, gradientBox.second);
    builder.append(", "_s);
    serializationForCSS(builder, gradientBox.secondRadius);
}

void Serialize<DeprecatedRadialGradient>::operator()(StringBuilder& builder, const DeprecatedRadialGradient& gradient)
{
    builder.append("radial, "_s);

    serializationForCSS(builder, gradient.gradientBox);

    if (!gradient.stops.isEmpty()) {
        builder.append(", "_s);
        serializationForCSS(builder, gradient.stops);
    }
}

// MARK: - ConicGradient

void Serialize<ConicGradient::GradientBox>::operator()(StringBuilder& builder, const ConicGradient::GradientBox& gradientBox)
{
    bool wroteSomething = false;

    if (gradientBox.angle) {
        WTF::switchOn(*gradientBox.angle,
            [&](const Angle<>::Raw& angleRaw) {
                if (angleRaw.value) {
                    builder.append("from "_s);
                    serializationForCSS(builder, angleRaw);
                    wroteSomething = true;
                }
            },
            [&](const Angle<>::Calc& angleCalc) {
                builder.append("from "_s);
                serializationForCSS(builder, angleCalc);
                wroteSomething = true;
            }
        );
    }

    if (gradientBox.position && !isCenterPosition(*gradientBox.position)) {
        if (wroteSomething)
            builder.append(' ');
        builder.append("at "_s);
        serializationForCSS(builder, *gradientBox.position);
    }
}

void Serialize<ConicGradient>::operator()(StringBuilder& builder, const ConicGradient& gradient)
{
    auto lengthBefore = builder.length();
    serializationForCSS(builder, gradient.gradientBox);
    bool wroteSomething = builder.length() != lengthBefore;

    if (appendColorInterpolationMethod(builder, gradient.colorInterpolationMethod, wroteSomething))
        wroteSomething = true;

    if (wroteSomething)
        builder.append(", "_s);

    serializationForCSS(builder, gradient.stops);
}

} // namespace CSS
} // namespace WebCore
