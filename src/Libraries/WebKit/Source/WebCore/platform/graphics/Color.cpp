/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "Color.h"

#include "ColorLuminance.h"
#include "ColorSerialization.h"
#include <cmath>
#include <wtf/Assertions.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Color);

static constexpr auto lightenedBlack = SRGBA<uint8_t> { 84, 84, 84 };
static constexpr auto darkenedWhite = SRGBA<uint8_t> { 171, 171, 171 };

Color::Color(const Color& other)
    : m_colorAndFlags(other.m_colorAndFlags)
{
    if (isOutOfLine())
        asOutOfLine().ref();
}

Color::Color(Color&& other)
{
    *this = WTFMove(other);
}

Color::Color(std::optional<ColorDataForIPC>&& colorData)
{
    if (colorData) {
        OptionSet<FlagsIncludingPrivate> flags;
        if (colorData->isSemantic)
            flags.add(FlagsIncludingPrivate::Semantic);
        if (colorData->usesFunctionSerialization)
            flags.add(FlagsIncludingPrivate::UseColorFunctionSerialization);

        WTF::switchOn(colorData->data,
            [&] (const PackedColor::RGBA& d) { setColor(asSRGBA(d), flags); },
            [&] (const OutOfLineColorDataForIPC& d) {
                setOutOfLineComponents(OutOfLineComponents::create({ d.c1, d.c2, d.c3, d.alpha }), d.colorSpace, flags);
            }
        );
    }
}

std::optional<ColorDataForIPC> Color::data() const
{
    if (!isValid())
        return std::nullopt;

    if (isOutOfLine()) {
        auto& outOfLineComponents = asOutOfLine();
        auto [c1, c2, c3, alpha] = outOfLineComponents.unresolvedComponents();

        OutOfLineColorDataForIPC oolcd = {
            .colorSpace = colorSpace(),
            .c1 = c1,
            .c2 = c2,
            .c3 = c3,
            .alpha = alpha
        };

        return { {
            .isSemantic = flags().contains(FlagsIncludingPrivate::Semantic),
            .usesFunctionSerialization = flags().contains(FlagsIncludingPrivate::UseColorFunctionSerialization),
            .data = oolcd
        } };

    } else {
        return { {
            .isSemantic = flags().contains(FlagsIncludingPrivate::Semantic),
            .usesFunctionSerialization = flags().contains(FlagsIncludingPrivate::UseColorFunctionSerialization),
            .data = asPackedInline()
        } };
    };
};

Color& Color::operator=(const Color& other)
{
    if (*this == other)
        return *this;

    if (isOutOfLine())
        asOutOfLine().deref();

    m_colorAndFlags = other.m_colorAndFlags;

    if (isOutOfLine())
        asOutOfLine().ref();

    return *this;
}

Color& Color::operator=(Color&& other)
{
    if (*this == other)
        return *this;

    if (isOutOfLine())
        asOutOfLine().deref();

    m_colorAndFlags = other.m_colorAndFlags;
    other.m_colorAndFlags = invalidColorAndFlags;

    return *this;
}

Color Color::lightened() const
{
    // Hardcode this common case for speed.
    if (isInline() && asInline() == black)
        return lightenedBlack;

    auto [r, g, b, a] = toColorTypeLossy<SRGBA<float>>().resolved();
    float v = std::max({ r, g, b });

    if (v == 0.0f)
        return lightenedBlack.colorWithAlphaByte(alphaByte());

    float multiplier = std::min(1.0f, v + 0.33f) / v;

    return convertColor<SRGBA<uint8_t>>(SRGBA<float> { multiplier * r, multiplier * g, multiplier * b, a });
}

Color Color::darkened() const
{
    // Hardcode this common case for speed.
    if (isInline() && asInline() == white)
        return darkenedWhite;
    
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<float>>().resolved();

    float v = std::max({ r, g, b });
    float multiplier = std::max(0.0f, (v - 0.33f) / v);

    return convertColor<SRGBA<uint8_t>>(SRGBA<float> { multiplier * r, multiplier * g, multiplier * b, a });
}

double Color::lightness() const
{
    // FIXME: Replace remaining uses with luminance.
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<float>>().resolved();
    auto [min, max] = std::minmax({ r, g, b });
    return 0.5 * (max + min);
}

double Color::luminance() const
{
    return WebCore::relativeLuminance(*this);
}

bool Color::anyComponentIsNone() const
{
    return callOnUnderlyingType([&]<typename ColorType> (const ColorType& underlyingColor) {
        if constexpr (std::is_same_v<ColorType, SRGBA<uint8_t>>)
            return false;
        else
            return underlyingColor.unresolved().anyComponentIsNone();
    });
}

Color Color::colorWithAlpha(float alpha) const
{
    return callOnUnderlyingType([&] (const auto& underlyingColor) -> Color {
        auto result = colorWithOverriddenAlpha(underlyingColor, alpha);

        // FIXME: Why is preserving the semantic bit desired and/or correct here?
        if (isSemantic())
            return { result, Flags::Semantic };

        return { result };
    });
}

Color Color::invertedColorWithAlpha(float alpha) const
{
    return callOnUnderlyingType([&]<typename ColorType> (const ColorType& underlyingColor) -> Color {
        // FIXME: Determine if there is a meaningful understanding of inversion that works
        // better for non-invertible color types like Lab or consider removing this in favor
        // of alternatives.
        if constexpr (ColorType::Model::isInvertible)
            return invertedColorWithOverriddenAlpha(underlyingColor, alpha);
        else
            return invertedColorWithOverriddenAlpha(convertColor<SRGBA<float>>(underlyingColor), alpha);
    });
}

Color Color::semanticColor() const
{
    if (isSemantic())
        return *this;
    
    if (isOutOfLine())
        return { asOutOfLineRef(), colorSpace(), Flags::Semantic };
    return { asInline(), Flags::Semantic };
}

ColorComponents<float, 4> Color::toResolvedColorComponentsInColorSpace(ColorSpace outputColorSpace) const
{
    auto [inputColorSpace, components] = colorSpaceAndResolvedColorComponents();
    return convertAndResolveColorComponents(inputColorSpace, components, outputColorSpace);
}

ColorComponents<float, 4> Color::toResolvedColorComponentsInColorSpace(const DestinationColorSpace& outputColorSpace) const
{
    auto [inputColorSpace, components] = colorSpaceAndResolvedColorComponents();
    return convertAndResolveColorComponents(inputColorSpace, components, outputColorSpace);
}

std::pair<ColorSpace, ColorComponents<float, 4>> Color::colorSpaceAndResolvedColorComponents() const
{
    if (isOutOfLine())
        return { colorSpace(), resolveColorComponents(asOutOfLine().resolvedComponents()) };
    return { ColorSpace::SRGB, asColorComponents(convertColor<SRGBA<float>>(asInline()).resolved()) };
}

bool Color::isBlackColor(const Color& color)
{
    return color.callOnUnderlyingType([] (const auto& underlyingColor) {
        return WebCore::isBlack(underlyingColor);
    });
}

bool Color::isWhiteColor(const Color& color)
{
    return color.callOnUnderlyingType([] (const auto& underlyingColor) {
        return WebCore::isWhite(underlyingColor);
    });
}

Color::DebugRGBA Color::debugRGBA() const
{
    auto [r, g, b, a] = toColorTypeLossy<SRGBA<uint8_t>>().resolved();
    return { r, g, b, a };
}

String Color::debugDescription() const
{
    return serializationForRenderTreeAsText(*this);
}

TextStream& operator<<(TextStream& ts, const Color& color)
{
    return ts << serializationForRenderTreeAsText(color);
}

} // namespace WebCore
