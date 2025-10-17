/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include "CSSGradientValue.h"

#include "CSSPrimitiveNumericTypes+CSSValueVisitation.h"
#include "CSSPrimitiveNumericTypes+Serialization.h"
#include "ColorInterpolation.h"
#include "StyleBuilderState.h"
#include "StyleGradientImage.h"
#include "StylePrimitiveNumericTypes+Conversions.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {
namespace {

template<typename> struct StyleImageIsUncacheable;

template<typename CSSType> static bool styleImageIsUncacheable(const CSSType& value)
{
    return StyleImageIsUncacheable<CSSType>()(value);
}

template<> struct StyleImageIsUncacheable<GradientColorInterpolationMethod> {
    constexpr bool operator()(const auto&) { return false; }
};

template<> struct StyleImageIsUncacheable<Color> {
    bool operator()(const auto& value) { return containsCurrentColor(value) || containsColorSchemeDependentColor(value); }
};

template<CSSValueID C> struct StyleImageIsUncacheable<Constant<C>> {
    constexpr bool operator()(const auto&) { return false; }
};

template<UnitEnum CSSType> struct StyleImageIsUncacheable<CSSType> {
    constexpr bool operator()(const auto& value) { return conversionToCanonicalUnitRequiresConversionData(value); }
};

template<NumericRaw CSSType> struct StyleImageIsUncacheable<CSSType> {
    constexpr bool operator()(const auto& value) { return styleImageIsUncacheable(value.unit); }
};

template<Calc CSSType> struct StyleImageIsUncacheable<CSSType> {
    constexpr bool operator()(const auto& value) { return value.protectedCalc()->requiresConversionData(); }
};

template<OptionalLike CSSType> struct StyleImageIsUncacheable<CSSType> {
    bool operator()(const auto& value) { return value && styleImageIsUncacheable(*value); }
};

template<TupleLike CSSType> struct StyleImageIsUncacheable<CSSType> {
    bool operator()(const auto& value) { return WTF::apply([&](const auto& ...x) { return (styleImageIsUncacheable(x) || ...); }, value); }
};

template<RangeLike CSSType> struct StyleImageIsUncacheable<CSSType> {
    bool operator()(const auto& value) { return std::ranges::any_of(value, [](auto& element) { return styleImageIsUncacheable(element); }); }
};

template<VariantLike CSSType> struct StyleImageIsUncacheable<CSSType> {
    bool operator()(const auto& value) { return WTF::switchOn(value, [](const auto& alternative) { return styleImageIsUncacheable(alternative); }); }
};

} // namespace (anonymous)
} // namespace CSS

// MARK: -

RefPtr<StyleImage> CSSGradientValue::createStyleImage(const Style::BuilderState& state) const
{
    if (m_cachedStyleImage)
        return m_cachedStyleImage;

    auto styleImage = StyleGradientImage::create(
        Style::toStyle(m_gradient, state)
    );
    if (!CSS::styleImageIsUncacheable(m_gradient))
        m_cachedStyleImage = styleImage.ptr();

    return styleImage;
}

String CSSGradientValue::customCSSText() const
{
    return CSS::serializationForCSS(m_gradient);
}

bool CSSGradientValue::equals(const CSSGradientValue& other) const
{
    return m_gradient == other.m_gradient;
}

IterationStatus CSSGradientValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    return CSS::visitCSSValueChildren(func, m_gradient);
}

} // namespace WebCore
