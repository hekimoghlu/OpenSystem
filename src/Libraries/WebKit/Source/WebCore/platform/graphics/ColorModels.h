/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#pragma once

#include <array>
#include <limits>
#include <optional>

namespace WebCore {

enum class RGBBoundedness { Bounded, Extended };

template<typename, RGBBoundedness> struct RGBModel;
template<typename> struct AlphaTraits;
template<typename> struct ColorComponentInfo;
template<typename> struct HSLModel;
template<typename> struct HWBModel;
template<typename> struct LCHModel;
template<typename> struct LabModel;
template<typename> struct OKLCHModel;
template<typename> struct OKLabModel;
template<typename> struct XYZModel;

// MARK: Resolved/Unresolved Definitions

template<typename ColorType, typename ColorModel = typename ColorType::Model> struct ExposedColorType;
template<typename ColorType> struct ResolvedColorType;
template<typename ColorType> struct UnresolvedColorType;

template<typename ColorType> constexpr ResolvedColorType<ColorType> resolvedColor(ColorType input)
{
    return ResolvedColorType<ColorType> { input };
}

template<typename ColorType> constexpr UnresolvedColorType<ColorType> unresolvedColor(ColorType input)
{
    return UnresolvedColorType<ColorType> { input };
}

inline constexpr ColorComponents<uint8_t, 4> resolveColorComponents(const ColorComponents<uint8_t, 4>& colorComponents)
{
    return colorComponents;
}

inline constexpr ColorComponents<float, 4> resolveColorComponents(const ColorComponents<float, 4>& colorComponents)
{
    return colorComponents.map([] (float component) { return std::isnan(component) ? 0.0f : component; });
}


template<typename ColorType> struct ResolvedColorType : ExposedColorType<ColorType, typename ColorType::Model> {
    using CanonicalType = ColorType;

    // Calling resolved() or unresolved() on a type that is already resolved is a no-op, so we can
    // just return ourselves.
    constexpr auto resolved() const { return *this; }
    constexpr auto unresolved() const { return *this; }

private:
    template<typename C> friend constexpr ResolvedColorType<C> resolvedColor(C);

    explicit constexpr ResolvedColorType(ColorType color)
        : ExposedColorType<ColorType, typename ColorType::Model> { resolve(color) }
    {
    }

    template<typename C, typename std::enable_if_t<std::is_same_v<typename C::ComponentType, float>>* = nullptr>
    static constexpr C resolve(C color)
    {
        auto [c1, c2, c3, alpha] = resolveColorComponents(asColorComponents(ExposedColorType<C, typename C::Model> { color }));
        return ColorType { c1, c2, c3, alpha };
    }

    template<typename C, typename std::enable_if_t<std::is_same_v<typename C::ComponentType, uint8_t>>* = nullptr>
    static constexpr C resolve(C color)
    {
        return color;
    }
};

template<typename ColorType> struct UnresolvedColorType : ExposedColorType<ColorType, typename ColorType::Model> {
    using CanonicalType = ColorType;

    // Calling unresolved() on a type that is already unresolved is a no-op, so we can
    // just return ourselves.
    constexpr auto unresolved() const { return *this; }

    constexpr bool anyComponentIsNone() const
    {
        auto [c1, c2, c3, alpha] = *this;
        return std::isnan(c1) || std::isnan(c2) || std::isnan(c3) || std::isnan(alpha);
    }

private:
    template<typename C> friend constexpr UnresolvedColorType<C> unresolvedColor(C);

    explicit constexpr UnresolvedColorType(ColorType color)
        : ExposedColorType<ColorType, typename ColorType::Model> { color }
    {
    }
};


// MARK: - Color Model support types.

template<> struct AlphaTraits<float> {
    static constexpr float transparent = 0.0f;
    static constexpr float opaque = 1.0f;
};

template<> struct AlphaTraits<uint8_t> {
    static constexpr uint8_t transparent = 0;
    static constexpr uint8_t opaque = 255;
};

// Analogous components categories as defined by CSS Color 4 - https://drafts.csswg.org/css-color-4/#analogous-components
enum class ColorComponentCategory {
    Reds,           // red (RGBModel), x (XYZModel)
    Greens,         // green (RGBModel), y (XYZModel)
    Blues,          // blue (RGBModel), z (XYZModel)
    Lightness,      // lightness (LCHModel, LabModel, OKLCHModel, OKLabModel)
    Colorfulness,   // chroma (LCHModel, OKLCHModel), saturation (HSLModel)
    Hue,            // hue (LCHModel, OKLCHModel, HSLModel, HWBModel)
    OpponentA,      // a (LabModel, OKLabModel)
    OpponentB       // b (LabModel, OKLabModel)
};

enum class ColorComponentType {
    Angle,
    Number
};

enum class ColorSpaceCoordinateSystem {
    RectangularOrthogonal,
    CylindricalPolar
};

template<typename T> struct ColorComponentInfo {
    T min;
    T max;
    ColorComponentType type;
    std::optional<ColorComponentCategory> category;
};

// MARK: - Color Model Definitions

// MARK: HSLModel

template<> struct HSLModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 360, ColorComponentType::Angle, ColorComponentCategory::Hue },
        { 0, std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Colorfulness },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Lightness }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::CylindricalPolar;
};

template<typename ColorType> struct ExposedColorType<ColorType, HSLModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::hue;
    using ColorType::saturation;
    using ColorType::lightness;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, HSLModel<T>>& c)
{
    return { c.hue, c.saturation, c.lightness, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesHSLModel = std::is_same_v<typename ColorType::Model, HSLModel<typename ColorType::ComponentType>>;

// MARK: HWBModel

template<> struct HWBModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 360, ColorComponentType::Angle, ColorComponentCategory::Hue },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, std::nullopt },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, std::nullopt }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::CylindricalPolar;
};

template<typename ColorType> struct ExposedColorType<ColorType, HWBModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::hue;
    using ColorType::whiteness;
    using ColorType::blackness;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, HWBModel<T>>& c)
{
    return { c.hue, c.whiteness, c.blackness, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesHWBModel = std::is_same_v<typename ColorType::Model, HWBModel<typename ColorType::ComponentType>>;

// MARK: LabModel

template<> struct LabModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 100, ColorComponentType::Number, ColorComponentCategory::Lightness },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::OpponentA },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::OpponentB }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<typename ColorType> struct ExposedColorType<ColorType, LabModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::lightness;
    using ColorType::a;
    using ColorType::b;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, LabModel<T>>& c)
{
    return { c.lightness, c.a, c.b, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesLabModel = std::is_same_v<typename ColorType::Model, LabModel<typename ColorType::ComponentType>>;

// MARK: LCHModel

template<> struct LCHModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 100, ColorComponentType::Number, ColorComponentCategory::Lightness },
        { 0, std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Colorfulness },
        { 0, 360, ColorComponentType::Angle, ColorComponentCategory::Hue }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::CylindricalPolar;
};

template<typename ColorType> struct ExposedColorType<ColorType, LCHModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::lightness;
    using ColorType::chroma;
    using ColorType::hue;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, LCHModel<T>>& c)
{
    return { c.lightness, c.chroma, c.hue, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesLCHModel = std::is_same_v<typename ColorType::Model, LCHModel<typename ColorType::ComponentType>>;

// MARK: OKLabModel

template<> struct OKLabModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 1, ColorComponentType::Number, ColorComponentCategory::Lightness },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::OpponentA },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::OpponentB }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<typename ColorType> struct ExposedColorType<ColorType, OKLabModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::lightness;
    using ColorType::a;
    using ColorType::b;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, OKLabModel<T>>& c)
{
    return { c.lightness, c.a, c.b, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesOKLabModel = std::is_same_v<typename ColorType::Model, OKLabModel<typename ColorType::ComponentType>>;

// MARK: OKLCHModel

template<> struct OKLCHModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 1, ColorComponentType::Number, ColorComponentCategory::Lightness },
        { 0, std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Colorfulness },
        { 0, 360, ColorComponentType::Angle, ColorComponentCategory::Hue }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::CylindricalPolar;
};

template<typename ColorType> struct ExposedColorType<ColorType, OKLCHModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::lightness;
    using ColorType::chroma;
    using ColorType::hue;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, OKLCHModel<T>>& c)
{
    return { c.lightness, c.chroma, c.hue, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesOKLCHModel = std::is_same_v<typename ColorType::Model, OKLCHModel<typename ColorType::ComponentType>>;

// MARK: RGBModel

template<> struct RGBModel<float, RGBBoundedness::Bounded> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { 0, 1, ColorComponentType::Number, ColorComponentCategory::Reds },
        { 0, 1, ColorComponentType::Number, ColorComponentCategory::Greens },
        { 0, 1, ColorComponentType::Number, ColorComponentCategory::Blues }
    } };
    static constexpr bool isInvertible = true;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<> struct RGBModel<uint8_t, RGBBoundedness::Bounded> {
    static constexpr std::array<ColorComponentInfo<uint8_t>, 3> componentInfo { {
        { 0, 255, ColorComponentType::Number, ColorComponentCategory::Reds },
        { 0, 255, ColorComponentType::Number, ColorComponentCategory::Greens },
        { 0, 255, ColorComponentType::Number, ColorComponentCategory::Blues }
    } };
    static constexpr bool isInvertible = true;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<> struct RGBModel<float, RGBBoundedness::Extended> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Reds },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Greens },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Blues }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<typename ColorType, RGBBoundedness boundedness> struct ExposedColorType<ColorType, RGBModel<typename ColorType::ComponentType, boundedness>> : ColorType {
    using ColorType::red;
    using ColorType::green;
    using ColorType::blue;
    using ColorType::alpha;
};

template<typename T, typename ColorType, RGBBoundedness boundedness> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, RGBModel<T, boundedness>>& c)
{
    return { c.red, c.green, c.blue, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesRGBModel =
       std::is_same_v<typename ColorType::Model, RGBModel<typename ColorType::ComponentType, RGBBoundedness::Bounded>>
    || std::is_same_v<typename ColorType::Model, RGBModel<typename ColorType::ComponentType, RGBBoundedness::Extended>>;


// MARK: XYZModel

template<> struct XYZModel<float> {
    static constexpr std::array<ColorComponentInfo<float>, 3> componentInfo { {
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Reds },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Greens },
        { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), ColorComponentType::Number, ColorComponentCategory::Blues }
    } };
    static constexpr bool isInvertible = false;
    static constexpr auto coordinateSystem = ColorSpaceCoordinateSystem::RectangularOrthogonal;
};

template<typename ColorType> struct ExposedColorType<ColorType, XYZModel<typename ColorType::ComponentType>> : ColorType {
    using ColorType::x;
    using ColorType::y;
    using ColorType::z;
    using ColorType::alpha;
};

template<typename T, typename ColorType> constexpr ColorComponents<T, 4> asColorComponents(const ExposedColorType<ColorType, XYZModel<T>>& c)
{
    return { c.x, c.y, c.z, c.alpha };
}

template<typename ColorType> inline constexpr bool UsesXYZModel = std::is_same_v<typename ColorType::Model, XYZModel<typename ColorType::ComponentType>>;


// get<> overload (along with std::tuple_size and std::tuple_element below) to support destructuring of explicitly resolved and unresolved colors.

template<size_t I, typename ColorType> constexpr typename ColorType::ComponentType get(const ExposedColorType<ColorType>& color)
{
    return asColorComponents(color)[I];
}

}

namespace std {

template<typename ColorType>
class tuple_size<WebCore::ResolvedColorType<ColorType>> : public std::integral_constant<size_t, 4> { };

template<size_t I, typename ColorType>
class tuple_element<I, WebCore::ResolvedColorType<ColorType>> {
public:
    using type = typename WebCore::ResolvedColorType<ColorType>::ComponentType;
};

template<typename ColorType>
class tuple_size<WebCore::UnresolvedColorType<ColorType>> : public std::integral_constant<size_t, 4> { };

template<size_t I, typename ColorType>
class tuple_element<I, WebCore::UnresolvedColorType<ColorType>> {
public:
    using type = typename WebCore::UnresolvedColorType<ColorType>::ComponentType;
};

}
