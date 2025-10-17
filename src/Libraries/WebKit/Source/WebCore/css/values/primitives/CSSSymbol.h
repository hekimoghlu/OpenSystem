/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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

#include "CSSValueTypes.h"

namespace WebCore {
namespace CSS {

struct SymbolRaw {
    CSSValueID value;

    constexpr bool operator==(const SymbolRaw&) const = default;
};

struct Symbol {
    using Raw = SymbolRaw;

    CSSValueID value;

    constexpr Symbol(SymbolRaw&& value)
        : value { value.value }
    {
    }

    constexpr Symbol(const SymbolRaw& value)
        : value { value.value }
    {
    }

    constexpr bool operator==(const Symbol&) const = default;
};

template<typename T> struct IsSymbol : public std::integral_constant<bool, std::is_same_v<T, Symbol>> { };

template<> struct Serialize<SymbolRaw> { void operator()(StringBuilder&, const SymbolRaw&); };
template<> struct Serialize<Symbol> { void operator()(StringBuilder&, const Symbol&); };

template<> struct ComputedStyleDependenciesCollector<SymbolRaw> { constexpr void operator()(ComputedStyleDependencies&, const SymbolRaw&) { } };
template<> struct ComputedStyleDependenciesCollector<Symbol> { constexpr void operator()(ComputedStyleDependencies&, const Symbol&) { } };

template<> struct CSSValueChildrenVisitor<SymbolRaw> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const SymbolRaw&) { return IterationStatus::Continue; } };
template<> struct CSSValueChildrenVisitor<Symbol> { constexpr IterationStatus operator()(const Function<IterationStatus(CSSValue&)>&, const Symbol&) { return IterationStatus::Continue; } };

} // namespace CSS
} // namespace WebCore
