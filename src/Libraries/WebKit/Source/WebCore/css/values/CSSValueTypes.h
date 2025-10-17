/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

#include "CSSValueAggregates.h"
#include "ComputedStyleDependencies.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

class CSSValue;

namespace CSS {

// MARK: - Serialization

// All leaf types must implement the following conversions:
//
//    template<> struct WebCore::CSS::Serialize<CSSType> {
//        void operator()(StringBuilder&, const CSSType&);
//    };

template<typename CSSType> struct Serialize;

// Serialization Invokers
template<typename CSSType> void serializationForCSS(StringBuilder& builder, const CSSType& value)
{
    Serialize<CSSType>{}(builder, value);
}

template<typename CSSType> [[nodiscard]] String serializationForCSS(const CSSType& value)
{
    StringBuilder builder;
    serializationForCSS(builder, value);
    return builder.toString();
}

template<typename CSSType> void serializationForCSSOnOptionalLike(StringBuilder& builder, const CSSType& value)
{
    if (!value)
        return;
    serializationForCSS(builder, *value);
}

template<typename CSSType> void serializationForCSSOnTupleLike(StringBuilder& builder, const CSSType& value, ASCIILiteral separator)
{
    auto swappedSeparator = ""_s;
    auto caller = WTF::makeVisitor(
        [&]<typename T>(const std::optional<T>& element) {
            if (!element)
                return;
            builder.append(std::exchange(swappedSeparator, separator));
            serializationForCSS(builder, *element);
        },
        [&]<typename T>(const Markable<T>& element) {
            if (!element)
                return;
            builder.append(std::exchange(swappedSeparator, separator));
            serializationForCSS(builder, *element);
        },
        [&](const auto& element) {
            builder.append(std::exchange(swappedSeparator, separator));
            serializationForCSS(builder, element);
        }
    );

    WTF::apply([&](const auto& ...x) { (..., caller(x)); }, value);
}

template<typename CSSType> void serializationForCSSOnRangeLike(StringBuilder& builder, const CSSType& value, ASCIILiteral separator)
{
    auto swappedSeparator = ""_s;
    for (const auto& element : value) {
        builder.append(std::exchange(swappedSeparator, separator));
        serializationForCSS(builder, element);
    }
}

template<typename CSSType> void serializationForCSSOnVariantLike(StringBuilder& builder, const CSSType& value)
{
    WTF::switchOn(value, [&](const auto& alternative) { serializationForCSS(builder, alternative); });
}

// Constrained for `TreatAsEmptyLike`.
template<EmptyLike CSSType> struct Serialize<CSSType> {
    void operator()(StringBuilder&, const CSSType&)
    {
    }
};

// Constrained for `TreatAsOptionalLike`.
template<OptionalLike CSSType> struct Serialize<CSSType> {
    void operator()(StringBuilder& builder, const CSSType& value)
    {
        serializationForCSSOnOptionalLike(builder, value);
    }
};

// Constrained for `TreatAsTupleLike`.
template<TupleLike CSSType> struct Serialize<CSSType> {
    void operator()(StringBuilder& builder, const CSSType& value)
    {
        serializationForCSSOnTupleLike(builder, value, SerializationSeparator<CSSType>);
    }
};

// Constrained for `TreatAsRangeLike`.
template<RangeLike CSSType> struct Serialize<CSSType> {
    void operator()(StringBuilder& builder, const CSSType& value)
    {
        serializationForCSSOnRangeLike(builder, value, SerializationSeparator<CSSType>);
    }
};

// Constrained for `TreatAsVariantLike`.
template<VariantLike CSSType> struct Serialize<CSSType> {
    void operator()(StringBuilder& builder, const CSSType& value)
    {
        serializationForCSSOnVariantLike(builder, value);
    }
};

// Specialization for `Constant`.
template<CSSValueID C> struct Serialize<Constant<C>> {
    void operator()(StringBuilder& builder, const Constant<C>& value)
    {
        builder.append(nameLiteralForSerialization(value.value));
    }
};

// Specialization for `CustomIdentifier`.
template<> struct Serialize<CustomIdentifier> {
    void operator()(StringBuilder&, const CustomIdentifier&);
};

// Specialization for `FunctionNotation`.
template<CSSValueID Name, typename CSSType> struct Serialize<FunctionNotation<Name, CSSType>> {
    void operator()(StringBuilder& builder, const FunctionNotation<Name, CSSType>& value)
    {
        builder.append(nameLiteralForSerialization(value.name), '(');
        serializationForCSS(builder, value.parameters);
        builder.append(')');
    }
};

// Specialization for `MinimallySerializingSpaceSeparatedRectEdges`.
template<typename CSSType> struct Serialize<MinimallySerializingSpaceSeparatedRectEdges<CSSType>> {
    void operator()(StringBuilder& builder, const MinimallySerializingSpaceSeparatedRectEdges<CSSType>& value)
    {
        constexpr auto separator = SerializationSeparator<MinimallySerializingSpaceSeparatedRectEdges<CSSType>>;

        if (value.left() != value.right()) {
            serializationForCSSOnTupleLike(builder, std::tuple { value.top(), value.right(), value.bottom(), value.left() }, separator);
            return;
        }
        if (value.bottom() != value.top()) {
            serializationForCSSOnTupleLike(builder, std::tuple { value.top(), value.right(), value.bottom() }, separator);
            return;
        }
        if (value.right() != value.top()) {
            serializationForCSSOnTupleLike(builder, std::tuple { value.top(), value.right() }, separator);
            return;
        }
        serializationForCSS(builder, value.top());
    }
};

// MARK: - Computed Style Dependencies

// What properties does this value rely on (eg, font-size for em units)?

// All non-tuple-like leaf types must implement the following conversions:
//
//    template<> struct WebCore::CSS::ComputedStyleDependenciesCollector<CSSType> {
//        void operator()(ComputedStyleDependencies&, const CSSType&);
//    };

template<typename CSSType> struct ComputedStyleDependenciesCollector;

// ComputedStyleDependencies Invoker
template<typename CSSType> void collectComputedStyleDependencies(ComputedStyleDependencies& dependencies, const CSSType& value)
{
    ComputedStyleDependenciesCollector<CSSType>{}(dependencies, value);
}

template<typename CSSType> [[nodiscard]] ComputedStyleDependencies collectComputedStyleDependencies(const CSSType& value)
{
    ComputedStyleDependencies dependencies;
    collectComputedStyleDependencies(dependencies, value);
    return dependencies;
}

template<typename CSSType> auto collectComputedStyleDependenciesOnOptionalLike(ComputedStyleDependencies& dependencies, const CSSType& value)
{
    if (!value)
        return;
    collectComputedStyleDependencies(dependencies, *value);
}

template<typename CSSType> auto collectComputedStyleDependenciesOnTupleLike(ComputedStyleDependencies& dependencies, const CSSType& value)
{
    WTF::apply([&](const auto& ...x) { (..., collectComputedStyleDependencies(dependencies, x)); }, value);
}

template<typename CSSType> auto collectComputedStyleDependenciesOnRangeLike(ComputedStyleDependencies& dependencies, const CSSType& value)
{
    for (const auto& element : value)
        collectComputedStyleDependencies(dependencies, element);
}

template<typename CSSType> auto collectComputedStyleDependenciesOnVariantLike(ComputedStyleDependencies& dependencies, const CSSType& value)
{
    WTF::switchOn(value, [&](const auto& alternative) { collectComputedStyleDependencies(dependencies, alternative); });
}

// Constrained for `TreatAsEmptyLike`.
template<EmptyLike CSSType> struct ComputedStyleDependenciesCollector<CSSType> {
    void operator()(ComputedStyleDependencies&, const CSSType&)
    {
    }
};

// Constrained for `TreatAsOptionalLike`.
template<OptionalLike CSSType> struct ComputedStyleDependenciesCollector<CSSType> {
    void operator()(ComputedStyleDependencies& dependencies, const CSSType& value)
    {
        collectComputedStyleDependenciesOnOptionalLike(dependencies, value);
    }
};

// Constrained for `TreatAsTupleLike`.
template<TupleLike CSSType> struct ComputedStyleDependenciesCollector<CSSType> {
    void operator()(ComputedStyleDependencies& dependencies, const CSSType& value)
    {
        collectComputedStyleDependenciesOnTupleLike(dependencies, value);
    }
};

// Constrained for `TreatAsRangeLike`.
template<RangeLike CSSType> struct ComputedStyleDependenciesCollector<CSSType> {
    void operator()(ComputedStyleDependencies& dependencies, const CSSType& value)
    {
        collectComputedStyleDependenciesOnRangeLike(dependencies, value);
    }
};

// Constrained for `TreatAsVariantLike`.
template<VariantLike CSSType> struct ComputedStyleDependenciesCollector<CSSType> {
    void operator()(ComputedStyleDependencies& dependencies, const CSSType& value)
    {
        collectComputedStyleDependenciesOnVariantLike(dependencies, value);
    }
};

// Specialization for `Constant`.
template<CSSValueID C> struct ComputedStyleDependenciesCollector<Constant<C>> {
    constexpr void operator()(ComputedStyleDependencies&, const Constant<C>&)
    {
        // Nothing to do.
    }
};

// Specialization for `CustomIdentifier`.
template<> struct ComputedStyleDependenciesCollector<CustomIdentifier> {
    constexpr void operator()(ComputedStyleDependencies&, const CustomIdentifier&)
    {
        // Nothing to do.
    }
};

// MARK: - CSSValue Visitation

// All non-tuple-like leaf types must implement the following conversions:
//
//    template<> struct WebCore::CSS::CSSValueChildrenVisitor<CSSType> {
//        IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const CSSType&);
//    };

template<typename CSSType> struct CSSValueChildrenVisitor;

// CSSValueVisitor Invoker
template<typename CSSType> IterationStatus visitCSSValueChildren(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
{
    return CSSValueChildrenVisitor<CSSType>{}(func, value);
}

template<typename CSSType> IterationStatus visitCSSValueChildrenOnOptionalLike(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
{
    return value ? visitCSSValueChildren(func, *value) : IterationStatus::Continue;
}

template<typename CSSType> IterationStatus visitCSSValueChildrenOnTupleLike(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
{
    // Process a single element of the tuple-like, updating result, and return true if result == IterationStatus::Done to
    // short circuit the fold in the apply lambda.
    auto process = [&](const auto& x, IterationStatus& result) -> bool {
        result = visitCSSValueChildren(func, x);
        return result == IterationStatus::Done;
    };

    return WTF::apply([&](const auto& ...x) {
        auto result = IterationStatus::Continue;
        (process(x, result) || ...);
        return result;
    }, value);
}

template<typename CSSType> IterationStatus visitCSSValueChildrenOnRangeLike(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
{
    for (const auto& element : value) {
        if (visitCSSValueChildren(func, element) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    return IterationStatus::Continue;
}

template<typename CSSType> IterationStatus visitCSSValueChildrenOnVariantLike(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
{
    return WTF::switchOn(value, [&](const auto& alternative) { return visitCSSValueChildren(func, alternative); });
}

// Constrained for `TreatAsEmptyLike`.
template<EmptyLike CSSType> struct CSSValueChildrenVisitor<CSSType> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const CSSType&)
    {
        return IterationStatus::Continue;
    }
};

// Constrained for `TreatAsOptionalLike`.
template<OptionalLike CSSType> struct CSSValueChildrenVisitor<CSSType> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
    {
        return visitCSSValueChildrenOnOptionalLike(func, value);
    }
};

// Constrained for `TreatAsTupleLike`.
template<TupleLike CSSType> struct CSSValueChildrenVisitor<CSSType> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
    {
        return visitCSSValueChildrenOnTupleLike(func, value);
    }
};

// Constrained for `TreatAsRangeLike`.
template<RangeLike CSSType> struct CSSValueChildrenVisitor<CSSType> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
    {
        return visitCSSValueChildrenOnRangeLike(func, value);
    }
};

// Constrained for `TreatAsVariantLike`.
template<VariantLike CSSType> struct CSSValueChildrenVisitor<CSSType> {
    IterationStatus operator()(const Function<IterationStatus(CSSValue&)>& func, const CSSType& value)
    {
        return visitCSSValueChildrenOnVariantLike(func, value);
    }
};

// Specialization for `Constant`.
template<CSSValueID C> struct CSSValueChildrenVisitor<Constant<C>> {
    constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const Constant<C>&)
    {
        return IterationStatus::Continue;
    }
};

// Specialization for `CustomIdentifier`.
template<> struct CSSValueChildrenVisitor<CustomIdentifier> {
    constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const CustomIdentifier&)
    {
        return IterationStatus::Continue;
    }
};

} // namespace CSS
} // namespace WebCore
