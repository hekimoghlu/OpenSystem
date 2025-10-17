/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "CSSColor.h"

#include "CSSAbsoluteColor.h"
#include "CSSColorLayers.h"
#include "CSSColorMix.h"
#include "CSSContrastColor.h"
#include "CSSLightDarkColor.h"
#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+ComputedStyleDependencies.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "CSSRelativeColor.h"
#include "StyleColorResolutionState.h"

namespace WebCore {
namespace CSS {

Color::Color(Color::ColorKind&& kind)
    : value { WTFMove(kind) }
{
}

Color::Color(Color::EmptyToken token)
    : value { token }
{
}

Color::Color(ResolvedColor&& color)
    : value { WTFMove(color) }
{
}

Color::Color(KeywordColor&& color)
    : value { WTFMove(color) }
{
}

Color::Color(HexColor&& color)
    : value { WTFMove(color) }
{
}

Color::Color(ColorLayers&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(ColorMix&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(ContrastColor&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(LightDarkColor&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<RGBFunctionLegacy<Number<>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<RGBFunctionLegacy<Percentage<>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<RGBFunctionModernAbsolute>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<HSLFunctionLegacy>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<HSLFunctionModern>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<HWBFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<LabFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<LCHFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<OKLabFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<OKLCHFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedA98RGB<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedDisplayP3<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedProPhotoRGB<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedRec2020<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedSRGBA<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorRGBFunction<ExtendedLinearSRGBA<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorXYZFunction<XYZA<float, WhitePoint::D50>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(AbsoluteColor<ColorXYZFunction<XYZA<float, WhitePoint::D65>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<RGBFunctionModernRelative>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<HSLFunctionModern>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<HWBFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<LabFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<LCHFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<OKLabFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<OKLCHFunction>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedA98RGB<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedDisplayP3<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedProPhotoRGB<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedRec2020<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedSRGBA<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorRGBFunction<ExtendedLinearSRGBA<float>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorXYZFunction<XYZA<float, WhitePoint::D50>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(RelativeColor<ColorXYZFunction<XYZA<float, WhitePoint::D65>>>&& color)
    : value { makeIndirectColor(WTFMove(color)) }
{
}

Color::Color(const Color& other)
    : value { copy(other.value) }
{
}

Color& Color::operator=(const Color& other)
{
    value = copy(other.value);
    return *this;
}

Color::Color(Color&&) = default;
Color& Color::operator=(Color&&) = default;
Color::~Color() = default;

bool Color::operator==(const Color& other) const
{
    if (value.index() != other.value.index())
        return false;

    return WTF::switchOn(value,
        [&](const EmptyToken&) {
            return true;
        },
        [&]<typename T>(const T& color) {
            return color == std::get<T>(other.value);
        },
        [&]<typename T>(const UniqueRef<T>& color) {
            return color.get() == std::get<UniqueRef<T>>(other.value).get();
        }
    );
}

bool Color::isResolved() const
{
    return std::holds_alternative<ResolvedColor>(value);
}

std::optional<ResolvedColor> Color::resolved() const
{
    if (isResolved())
        return std::get<ResolvedColor>(value);
    return std::nullopt;
}

bool Color::isKeyword() const
{
    return std::holds_alternative<KeywordColor>(value);
}

std::optional<KeywordColor> Color::keyword() const
{
    if (isKeyword())
        return std::get<KeywordColor>(value);
    return std::nullopt;
}

bool Color::isHex() const
{
    return std::holds_alternative<HexColor>(value);
}

std::optional<HexColor> Color::hex() const
{
    if (isHex())
        return std::get<HexColor>(value);
    return std::nullopt;
}

WebCore::Color Color::absoluteColor() const
{
    return switchOn(
        [](const ResolvedColor& resolved) -> WebCore::Color {
            return resolved.value;
        },
        [](const KeywordColor& keyword) -> WebCore::Color {
            if (isAbsoluteColorKeyword(keyword.valueID))
                return colorFromAbsoluteKeyword(keyword.valueID);
            return { };
        },
        [](const HexColor& hex) -> WebCore::Color {
            return WebCore::Color { hex.value };
        },
        [](const auto&) -> WebCore::Color {
            return { };
        }
    );
}

Color::ColorKind Color::copy(const Color::ColorKind& other)
{
    return WTF::switchOn(other,
        []<typename T>(const T& color) -> Color::ColorKind {
            return color;
        },
        []<typename T>(const UniqueRef<T>& color) -> Color::ColorKind {
            return makeUniqueRef<T>(color.get());
        }
    );
}

template<typename T> Color::ColorKind Color::makeIndirectColor(T&& color)
{
    return { makeUniqueRef<T>(WTFMove(color)) };
}

// MARK: - Markable Traits

bool Color::MarkableTraits::isEmptyValue(const Color& value)
{
    return std::holds_alternative<EmptyToken>(value.value);
}

Color Color::MarkableTraits::emptyValue()
{
    return Color(EmptyToken());
}

WebCore::Color createColor(const Color& value, PlatformColorResolutionState& state)
{
    return WTF::switchOn(value, [&](const auto& color) { return WebCore::CSS::createColor(color, state); });
}

bool containsCurrentColor(const Color& value)
{
    return WTF::switchOn(value, [&](const auto& color) { return WebCore::CSS::containsCurrentColor(color); });
}

bool containsColorSchemeDependentColor(const Color& value)
{
    return WTF::switchOn(value, [&](const auto& color) { return WebCore::CSS::containsColorSchemeDependentColor(color); });
}

void Serialize<Color>::operator()(StringBuilder& builder, const Color& value)
{
    WTF::switchOn(value, [&](const auto& color) { serializationForCSS(builder, color); });
}

void ComputedStyleDependenciesCollector<Color>::operator()(ComputedStyleDependencies&dependencies, const Color& value)
{
    WTF::switchOn(value, [&](const auto& color) { collectComputedStyleDependencies(dependencies, color); });
}

IterationStatus CSSValueChildrenVisitor<Color>::operator()(const Function<IterationStatus(CSSValue&)>& func, const Color& value)
{
    return WTF::switchOn(value, [&](const auto& color) { return visitCSSValueChildren(func, color); });
}

} // namespace CSS
} // namespace WebCore
